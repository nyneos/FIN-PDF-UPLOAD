#!/usr/bin/env python
"""
Test script for chunked extraction with streaming.
Tests the /parse/stream endpoint and validates chunked LLM calls.
"""
import requests
import json
import time
import sys

def test_stream_parsing():
    """Test the /parse/stream endpoint with a PDF file."""
    pdf_path = '/Users/hardikmishra/Downloads/bank_parser_full/Fw_ Sample Bank Account statements/544516929-ICICI-BANK-STATEMENT.pdf'
    url = 'http://localhost:8000/parse/stream'
    
    print("=" * 80)
    print("TESTING CHUNKED EXTRACTION WITH STREAMING")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        with open(pdf_path, 'rb') as f:
            files = {'pdf': ('test.pdf', f, 'application/pdf')}
            print(f"\n[*] Sending request to {url}...")
            resp = requests.post(url, files=files, stream=True, timeout=300)
            resp.raise_for_status()
            
            print(f"[*] Streaming response starting...\n")
            
            final_result = None
            phase_times = {}
            last_phase = None
            phase_start = start_time
            
            for line in resp.iter_lines():
                if not line:
                    continue
                    
                data = json.loads(line)
                status = data.get('status', 'unknown')
                latency = data.get('latency_ms', 0) / 1000.0  # Convert to seconds
                
                # Track phase transitions
                if status != last_phase:
                    if last_phase:
                        phase_times[last_phase] = latency
                    last_phase = status
                
                # Print progress
                if status == 'extraction_started':
                    print(f"[✓] Extraction started ({latency:.1f}s)")
                
                elif status == 'ocr_progress':
                    page = data.get('page', '?')
                    total = data.get('total_pages', '?')
                    lines = data.get('cumulative_ocr_lines', 0)
                    print(f"  [OCR] Page {page}/{total} - {lines} cumulative lines")
                
                elif status == 'ocr_complete':
                    lines = data.get('total_ocr_lines', 0)
                    print(f"[✓] OCR complete ({latency:.1f}s) - {lines} total lines")
                
                elif status == 'sending_to_ai':
                    layout = data.get('layout_lines', 0)
                    tables = data.get('table_rows', 0)
                    ocr = data.get('ocr_lines', 0)
                    print(f"[*] Sending to AI ({latency:.1f}s):")
                    print(f"    Layout: {layout} lines | Tables: {tables} rows | OCR: {ocr} lines")
                
                elif status == 'complete':
                    final_result = data
                    clean = data.get('clean', {})
                    debug = data.get('debug', {})
                    
                    transactions = clean.get('transactions', [])
                    llm_calls = debug.get('llm_calls_made', 1)
                    total_chars = debug.get('total_available_chars', 0)
                    
                    print(f"[✓] Extraction complete ({latency:.1f}s)")
                    print(f"\n    RESULTS:")
                    print(f"    ├─ Transactions: {len(transactions)}")
                    print(f"    ├─ LLM calls: {llm_calls}")
                    print(f"    ├─ Total available data: {total_chars:,} chars")
                    print(f"    └─ Opening balance: {clean.get('opening_balance', 'N/A')}")
                    
                    validation = clean.get('validation', {})
                    if validation:
                        print(f"\n    VALIDATION:")
                        print(f"    ├─ Running balance matches: {validation.get('matches', False)}")
                        print(f"    ├─ Balance variance: {validation.get('balance_variance', 'N/A')}")
                        print(f"    └─ Issues: {validation.get('issues', 0)}")
                
                elif status == 'error':
                    print(f"[✗] Error: {data.get('error', 'Unknown error')}")
                    return False
            
            elapsed = time.time() - start_time
            
            if final_result:
                print(f"\n" + "=" * 80)
                print("SUMMARY")
                print("=" * 80)
                print(f"Total latency: {elapsed:.1f}s")
                print(f"Phase breakdown:")
                for phase, t in phase_times.items():
                    print(f"  - {phase}: {t:.1f}s")
                
                # Show first few transactions
                transactions = final_result.get('clean', {}).get('transactions', [])
                if transactions:
                    print(f"\nFirst 3 transactions:")
                    for i, txn in enumerate(transactions[:3]):
                        print(f"  {i+1}. {txn.get('tran_date')} | {txn.get('narration', 'N/A')[:40]}")
                        print(f"     Debit: {txn.get('withdrawal')}, Credit: {txn.get('deposit')}, Balance: {txn.get('balance')}")
                
                return True
            else:
                print("[✗] No final result received")
                return False
                
    except requests.exceptions.RequestException as e:
        print(f"[✗] Request failed: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"[✗] JSON decode error: {e}")
        return False
    except Exception as e:
        print(f"[✗] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_stream_parsing()
    sys.exit(0 if success else 1)
