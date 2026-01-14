"""
AI Verification Layer:
Normalizes table rows into structured transactions using Groq.
SIMPLIFIED: Single-shot LLM call + local fallback (no multi-agent overhead).
"""

import json
import re
from groq_client import groq_llm
from extract.preprocess import flatten_layout
from logging_config import configure_logging
from config import config
from ai.prompt_memory import add_suggestion
from typing import List
from json import JSONDecodeError

logger = configure_logging()


def _join_table_rows(table_data) -> List[str]:
	rows = []
	for r in table_data:
		if isinstance(r, (list, tuple)):
			rows.append(" | ".join([str(c).strip() for c in r if c is not None]))
		else:
			rows.append(str(r))
	return rows


def _select_relevant_lines(lines: List[str], max_chars: int) -> List[str]:
	"""Select lines that look like transactions (contain digits/amounts)."""
	if not lines:
		return []

	candidates = [ln for ln in lines if re.search(r"\d", ln)]
	# fallback to top lines if no candidates
	if not candidates:
		candidates = lines

	selected = []
	total = 0
	for ln in candidates:
		if total + len(ln) > max_chars:
			break
		selected.append(ln)
		total += len(ln)

	return selected


def _truncate_payload(layout_lines, table_rows, ocr_lines, max_chars):
	"""Truncate the content while preserving likely transaction lines."""
	# prioritize table rows then layout lines then ocr
	tbl = _join_table_rows(table_rows)

	# attempt to collect relevant table rows first
	out = {"table": [], "layout": [], "ocr": [], "truncated": False}
	remaining = max_chars

	for r in tbl:
		if len(r) <= remaining:
			out["table"].append(r)
			remaining -= len(r)
		else:
			out["truncated"] = True
			break

	if remaining > 0:
		layout_sel = _select_relevant_lines(layout_lines, remaining)
		out["layout"].extend(layout_sel)
		remaining -= sum(len(x) for x in layout_sel)

	if remaining > 0:
		ocr_sel = _select_relevant_lines(ocr_lines, remaining)
		out["ocr"].extend(ocr_sel)
		remaining -= sum(len(x) for x in ocr_sel)

	return out


def _local_fallback_parse(table_rows, layout_lines, ocr_lines):
	"""Try to convert extracted rows into a list of transactions using regex heuristics."""
	"""
	Improved fallback parser that groups successive layout lines into transaction
	blocks. Heuristics:
	- Skip obvious header/footer lines (Page, Date, Particulars, Branch, IFSC, etc.)
	- When a date line is encountered, start a new transaction block.
	- Collect following lines as narration until an amount/balance line is found.
	- Parse consecutive numeric lines as [amount, balance] where possible.
	- Normalize dates to ISO (YYYY-MM-DD) when recognized.
	This provides much better results when `table_rows` is empty and layout text
	contains separate lines for date, narration and amounts.
	"""

	txns = []

	amount_re = re.compile(r"\d{1,3}(?:[,.]\d{3})*(?:[.,]\d{2})")
	# match dd-mm-yyyy, dd/mm/yyyy, yyyy-mm-dd
	date_re = re.compile(r"(\d{2}[\-/]\d{2}[\-/]\d{2,4}|\d{4}[\-/]\d{2}[\-/]\d{2})")

	# Normalize helper for dates
	from datetime import datetime

	def norm_date(s: str):
		s = s.strip()
		for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%Y-%m-%d", "%Y/%m/%d"):
			try:
				d = datetime.strptime(s, fmt)
				return d.strftime("%Y-%m-%d")
			except Exception:
				continue
		# try fuzzy two-part (like '01-05-2020') via split
		m = re.search(date_re, s)
		if m:
			token = m.group(0)
			for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
				try:
					d = datetime.strptime(token, fmt)
					return d.strftime("%Y-%m-%d")
				except Exception:
					continue
		return None

	# Build a working list of lines to parse. If table_rows present, prefer them,
	# otherwise use layout_lines.
	if table_rows:
		raw_rows = []
		for r in table_rows:
			if isinstance(r, (list, tuple)):
				raw_rows.append(" ".join([str(c).strip() for c in r if c is not None]))
			else:
				raw_rows.append(str(r).strip())
	else:
		raw_rows = [ln.strip() for ln in (layout_lines or []) if ln and ln.strip()]

	# filter helper
	header_blacklist = re.compile(r"^(page\b|date\b|particulars\b|instrument\b|withdrawals\b|deposits\b|balance\b|client code\b|branch code\b|ifsc code\b|branch name\b|address\b|phone\b|statement for a/c|page \d+ of|page\s*\d+)", re.I)

	i = 0
	last_balance = None
	while i < len(raw_rows):
		ln = raw_rows[i].strip()
		# skip headers/footers
		if not ln or header_blacklist.search(ln.lower()):
			i += 1
			continue

		# If this line contains an explicit date at start or isolated date, start txn
		date_m = date_re.search(ln)
		if date_m and (ln.startswith(date_m.group(0)) or len(ln) <= 30):
			tran_date = norm_date(date_m.group(0))
			narration_parts = []
			i += 1
			# collect narration lines until we hit amount-like lines
			while i < len(raw_rows):
				nxt = raw_rows[i].strip()
				if not nxt:
					i += 1
					continue
				# stop if next is another date or a page marker or header
				if date_re.search(nxt) and (nxt.startswith(date_re.search(nxt).group(0)) or len(nxt) <= 30):
					break
				if header_blacklist.search(nxt.lower()):
					i += 1
					continue
				# if we find an amount line, break to parse amounts
				if amount_re.search(nxt):
					break
				narration_parts.append(nxt)
				i += 1

			# now collect consecutive numeric/amount lines (likely amount then balance)
			amounts = []
			while i < len(raw_rows) and amount_re.search(raw_rows[i]):
				a_line = raw_rows[i]
				found = amount_re.findall(a_line)
				# take the last captured numeric token in the line
				if found:
					amounts.extend(found)
				i += 1

			# build transaction object
			narration = " ".join(narration_parts).strip() or None
			tran = {
				"tran_date": tran_date,
				"value_date": tran_date,
				"narration": narration,
				"withdrawal": None,
				"deposit": None,
				"balance": None,
			}

			# interpret amounts: last amount is most likely balance
			try:
				if amounts:
					# remove commas and convert
					cleaned = [float(a.replace(',', '')) for a in amounts]
					if len(cleaned) >= 2:
						# assume last is balance, previous is amount (deposit/withdrawal)
						tran["balance"] = cleaned[-1]
						amt = cleaned[-2]
						# Heuristic: if balance increased vs last_balance, treat as deposit
						if last_balance is not None and tran["balance"] is not None:
							if tran["balance"] > last_balance:
								tran["deposit"] = amt
							else:
								tran["withdrawal"] = amt
						else:
							# default to deposit if unclear
							tran["deposit"] = amt
						last_balance = tran.get("balance") or last_balance
					else:
						# single amount: treat as deposit by default, but if contains '-' treat withdrawal
						amt = cleaned[-1]
						if re.search(r"\bDR\b|\-", ' '.join(narration_parts), re.I):
							tran["withdrawal"] = amt
						else:
							tran["deposit"] = amt
				else:
					# no amounts found for this date block
					pass
			except Exception:
				pass

			txns.append(tran)
			continue

		# If current line itself contains amounts and no date was found, try to attach
		# it to the last transaction as amounts/balance
		if amount_re.search(ln) and txns:
			found = amount_re.findall(ln)
			try:
				cleaned = [float(a.replace(',', '')) for a in found]
				if len(cleaned) == 1:
					# attach as deposit if last tx has no amount
					last = txns[-1]
					if last.get("deposit") is None and last.get("withdrawal") is None:
						last["deposit"] = cleaned[0]
					else:
						# prefer updating balance
						last["balance"] = cleaned[0]
				elif len(cleaned) >= 2:
					last = txns[-1]
					last["deposit"] = cleaned[-2]
					last["balance"] = cleaned[-1]
			except Exception:
				pass
			i += 1
			continue

		# otherwise skip miscellaneous lines
		i += 1

	return txns


def _extract_json_from_text(text: str):
	"""Attempt to find and parse JSON embedded in LLM text."""
	text = text.strip()

	def remove_think_mode(t: str) -> str:
		t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE)
		t = t.replace("<think>", "").replace("</think>", "")
		return t.strip()

	text = remove_think_mode(text)

	# try direct parse first
	try:
		return json.loads(text)
	except JSONDecodeError:
		pass

	# fenced json ```json { ... } ```
	m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
	if m:
		try:
			return json.loads(m.group(1))
		except JSONDecodeError:
			pass

	# any ``` fence that contains JSON
	m = re.search(r"```\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
	if m:
		try:
			return json.loads(m.group(1))
		except JSONDecodeError:
			pass

	# best effort: find first { and last } and parse
	start = text.find("{")
	end = text.rfind("}")
	if start != -1 and end != -1 and end > start:
		candidate = text[start:end+1]
		try:
			return json.loads(candidate)
		except JSONDecodeError:
			pass

	# if lines contain a JSON object starting line, try from that line
	lines = text.splitlines()
	for idx, line in enumerate(lines):
		if line.strip().startswith("{") or line.strip().startswith("["):
			try:
				sub = "\n".join(lines[idx:])
				return json.loads(sub)
			except JSONDecodeError:
				continue

	raise JSONDecodeError("No JSON found", text, 0)


def _coerce_transaction_schema(rows):
	"""Coerce common AI output shapes into the required transaction schema."""
	out = []
	if not rows:
		return out

	for r in rows:
		if not isinstance(r, dict):
			continue
		item = {
			"tran_date": None,
			"value_date": None,
			"narration": None,
			"withdrawal": None,
			"deposit": None,
			"balance": None,
		}

		# Dates
		for k in ("tran_date", "value_date", "date", "txn_date", "transaction_date"):
			if k in r and r.get(k):
				item["tran_date"] = r.get(k)
				item["value_date"] = r.get(k)
				break

		# Narration / description
		for k in ("narration", "description", "particulars", "remarks"):
			if k in r and r.get(k):
				item["narration"] = r.get(k)
				break

		# Balance (common keys)
		for k in ("balance", "running_balance", "bal"):
			if k in r and r.get(k) is not None:
				item["balance"] = r.get(k)
				break

		def to_float(v):
			try:
				if isinstance(v, str):
					return float(v.replace(",", "").replace("₹", "").strip())
				return float(v)
			except Exception:
				return None

		# explicit keys
		if "withdrawal" in r and r.get("withdrawal") is not None:
			item["withdrawal"] = to_float(r.get("withdrawal"))
		if "deposit" in r and r.get("deposit") is not None:
			item["deposit"] = to_float(r.get("deposit"))

		# debit/credit columns
		if item["withdrawal"] is None and item["deposit"] is None:
			for dk in ("debit", "dr", "debit_amount"):
				if dk in r and r.get(dk):
					item["withdrawal"] = to_float(r.get(dk))
					break
			for ck in ("credit", "cr", "credit_amount"):
				if ck in r and r.get(ck):
					item["deposit"] = to_float(r.get(ck))
					break

		# fallback: single 'amount' + optional 'type'
		if item["withdrawal"] is None and item["deposit"] is None and "amount" in r and r.get("amount") is not None:
			amt = to_float(r.get("amount"))
			t = (r.get("type") or r.get("txn_type") or "").lower()
			if t in ("debit", "dr", "withdrawal"):
				item["withdrawal"] = amt
			elif t in ("credit", "cr", "deposit"):
				item["deposit"] = amt
			else:
				item["deposit"] = amt

		if isinstance(item.get("balance"), str):
			item["balance"] = to_float(item.get("balance"))

		out.append(item)

	return out


def _split_into_chunks(layout_lines, table_rows, ocr_lines, chunk_size=29000, num_chunks=4):
	"""
	Intelligently split extraction data into overlapping chunks to maximize transaction coverage.
	
	Strategy:
	- Distribute table_rows (most transaction-dense) across chunks evenly
	- Assign sequential OCR lines to each chunk (some overlap for context)
	- Allow chunks to overlap slightly (10% of chunk_size) to catch transactions at boundaries
	- Return list of chunk dicts with 'layout', 'table', 'ocr' keys
	"""
	chunks = []
	
	# Table rows: distribute evenly across chunks
	table_rows_per_chunk = max(1, len(table_rows) // num_chunks)
	ocr_lines_per_chunk = max(1, len(ocr_lines) // num_chunks)
	layout_lines_per_chunk = max(1, len(layout_lines) // num_chunks)
	
	overlap_chars = chunk_size // 10  # 10% overlap for boundary transactions
	
	for i in range(num_chunks):
		start_table = i * table_rows_per_chunk
		end_table = (i + 1) * table_rows_per_chunk if i < num_chunks - 1 else len(table_rows)
		
		# Add slight overlap to next chunk's start
		if i < num_chunks - 1:
			end_table = min(end_table + 2, len(table_rows))
		
		start_ocr = max(0, i * ocr_lines_per_chunk - 5)  # 5 line overlap
		end_ocr = (i + 1) * ocr_lines_per_chunk + 5 if i < num_chunks - 1 else len(ocr_lines)
		
		start_layout = max(0, i * layout_lines_per_chunk - 5)
		end_layout = (i + 1) * layout_lines_per_chunk + 5 if i < num_chunks - 1 else len(layout_lines)
		
		chunk = {
			"index": i,
			"table": table_rows[start_table:end_table],
			"ocr": ocr_lines[start_ocr:end_ocr],
			"layout": layout_lines[start_layout:end_layout],
		}
		chunks.append(chunk)
	
	logger.info(f"[AI] Split data into {len(chunks)} chunks (avg {chunk_size} chars each)")
	return chunks


def _deduplicate_transactions(all_transactions):
	"""
	Deduplicate transactions from multiple chunks using (tran_date, narration, deposit/withdrawal) as key.
	Prefers complete entries over partial ones.
	"""
	seen = {}
	unique = []
	
	for txn in all_transactions:
		if not txn.get("tran_date") or not txn.get("narration"):
			unique.append(txn)
			continue
		
		# Create key from date + narration + primary amount
		amt = txn.get("deposit") or txn.get("withdrawal") or 0
		key = (txn.get("tran_date"), txn.get("narration")[:50], round(float(amt) if amt else 0, 2))
		
		if key not in seen:
			seen[key] = txn
			unique.append(txn)
		else:
			# If this entry is more complete than the previous, replace it
			prev = seen[key]
			prev_score = sum(1 for v in [prev.get("deposit"), prev.get("withdrawal"), prev.get("balance")] if v is not None)
			curr_score = sum(1 for v in [txn.get("deposit"), txn.get("withdrawal"), txn.get("balance")] if v is not None)
			
			if curr_score > prev_score:
				# Find and replace in unique list
				idx = unique.index(prev)
				unique[idx] = txn
				seen[key] = txn
	
	logger.info(f"[AI] Deduplicated {len(all_transactions)} transactions to {len(unique)} unique entries")
	return unique


def _send_chunk_to_llm(chunk_data, chunk_index, total_chunks):
	"""Send a single chunk to LLM for extraction."""
	layout_lines = chunk_data.get("layout", [])
	table_rows = chunk_data.get("table", [])
	ocr_lines = chunk_data.get("ocr", [])
	
	# Build prompt for this chunk
	prompt_content = ""
	remaining = 28000  # Leave 1KB buffer per chunk
	
	# Add table rows first (most important)
	if table_rows and remaining > 500:
		tbl = _join_table_rows(table_rows)
		table_section = "\n".join(tbl)
		if len(table_section) <= remaining:
			prompt_content += "table_data:\n" + table_section + "\n\n"
			remaining -= len(table_section) + 12
		else:
			# Truncate table: first + last rows
			truncated_tbl = (tbl[:15] if len(tbl) > 30 else tbl[:10]) + ["..."] + (tbl[-10:] if len(tbl) > 30 else [])
			table_section = "\n".join(truncated_tbl)
			if len(table_section) <= remaining:
				prompt_content += "table_data:\n" + table_section + "\n\n"
				remaining -= len(table_section) + 12
	
	# Add OCR lines
	if ocr_lines and remaining > 500:
		ocr_selected = _select_relevant_lines(ocr_lines, remaining - 200)
		if ocr_selected:
			prompt_content += "ocr_data:\n" + "\n".join(ocr_selected) + "\n\n"
			remaining -= sum(len(ln) for ln in ocr_selected) + 12
	
	# Add layout lines only if space remains
	if layout_lines and remaining > 500:
		layout_selected = _select_relevant_lines(layout_lines, remaining - 100)
		if layout_selected:
			prompt_content += "layout_text:\n" + "\n".join(layout_selected)
			remaining -= sum(len(ln) for ln in layout_selected) + 12
	
	prompt_template = """
You are an expert financial statement parser.

Extract ALL transactions from this section (chunk {chunk}/{total}) of a bank statement.

RETURN ONLY THIS JSON (no text, no explanation):
{{
  "metadata": {{
    "account_number": null,
    "account_name": null,
    "bank_name": null,
    "ifsc": null,
    "micr": null,
    "period_start": null,
    "period_end": null,
    "opening_balance": null,
    "closing_balance": null
  }},
  "transactions": [
    {{
      "tran_date": "YYYY-MM-DD",
      "value_date": "YYYY-MM-DD",
      "narration": "string",
      "withdrawal": number or null,
      "deposit": number or null,
      "balance": number or null
    }}
  ]
}}

RULES:
- Dates: normalize to YYYY-MM-DD; if one date present, use for both tran_date and value_date
- Withdrawal (DR/Debit): positive float
- Deposit (CR/Credit): positive float
- Remove headers, summaries, balance lines, page markers
- Narration: transaction description only, no dates/amounts
- All numbers: plain floats, no commas
- Extract EVERY transaction visible in this section (chunk {chunk}/{total})

DATA:
<<PROMPT_CONTENT>>

RETURN ONLY JSON.
"""

	prompt = prompt_template.format(chunk=chunk_index + 1, total=total_chunks).replace('<<PROMPT_CONTENT>>', prompt_content)
	
	logger.info(f"[AI] Chunk {chunk_index + 1}/{total_chunks}: sending {len(prompt_content)} chars to LLM...")
	
	try:
		raw = groq_llm([
			{"role": "system", "content": "You are an expert bank statement parser. Return JSON only."},
			{"role": "user", "content": prompt},
		])
		raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
		raw = raw.replace("<think>", "").replace("</think>", "")
		
		try:
			parsed = _extract_json_from_text(raw)
		except JSONDecodeError:
			parsed = None
		
		if parsed and isinstance(parsed, dict):
			transactions = parsed.get("transactions", [])
			logger.info(f"[AI] Chunk {chunk_index + 1}/{total_chunks}: extracted {len(transactions)} transactions")
			return {
				"transactions": _coerce_transaction_schema(transactions),
				"metadata": parsed.get("metadata", {}),
			}
		
		if isinstance(parsed, list):
			logger.info(f"[AI] Chunk {chunk_index + 1}/{total_chunks}: extracted {len(parsed)} transactions (list)")
			return {
				"transactions": _coerce_transaction_schema(parsed),
				"metadata": {},
			}
	
	except Exception as e:
		logger.warning(f"[AI] Chunk {chunk_index + 1}/{total_chunks} failed: {e}")
	
	# Fallback for this chunk
	logger.info(f"[AI] Chunk {chunk_index + 1}/{total_chunks}: falling back to local parse")
	local_txns = _local_fallback_parse(table_rows, layout_lines, ocr_lines)
	return {
		"transactions": local_txns,
		"metadata": {},
	}


def verify_and_clean(layout_data, table_data, ocr_data):
	"""
	EMERGENCY MODE: Single-call LLM with smart truncation.
	Disables chunking to prevent exhausting daily token limits.
	Uses intelligent prioritization: table → layout → OCR.
	"""
	layout_lines = flatten_layout(layout_data)
	table_rows = table_data
	ocr_lines = ocr_data or []

	# Single-call mode: truncate intelligently
	truncated_data = _truncate_payload(layout_lines, table_rows, ocr_lines, 
	                                    max_chars=config.AI_MAX_PAYLOAD_CHARS)
	
	# Build payload with priority: table > layout > ocr
	parts = []
	if truncated_data["table"]:
		parts.append("=== TABLE ===\n" + "\n".join(truncated_data["table"]))
	if truncated_data["layout"]:
		parts.append("=== LAYOUT ===\n" + "\n".join(truncated_data["layout"]))
	if truncated_data["ocr"]:
		parts.append("=== OCR ===\n" + "\n".join(truncated_data["ocr"]))
	
	payload = "\n".join(parts)
	total_chars = len(payload)
	
	logger.info(f"[AI] Payload {total_chars} chars (truncated={truncated_data['truncated']}) - SINGLE CALL MODE")

	# Format as ChatCompletion messages array (not string)
	messages = [
		{
			"role": "user",
			"content": f"""Extract bank transactions from the following statement data.

Return a JSON array of transactions with this structure:
[{{"tran_date": "YYYY-MM-DD", "narration": "...", "withdrawal": 0.0, "deposit": 0.0, "balance": 0.0}}]

Only include fields that have data. Dates must be ISO format.

{payload}"""
		}
	]

	# Single LLM call - NO CHUNKING
	result = None
	try:
		result = groq_llm(messages)
	except Exception as e:
		logger.warning(f"[AI] LLM call failed: {e}")
	
	transactions = []
	if result:
		try:
			parsed = json.loads(result)
			if isinstance(parsed, list):
				transactions = parsed
			elif isinstance(parsed, dict):
				transactions = parsed.get("transactions", [])
		except (json.JSONDecodeError, AttributeError) as e:
			logger.warning(f"[AI] Failed to parse LLM response: {e}, using local fallback")
			transactions = _local_fallback_parse(table_rows, layout_lines, ocr_lines)
	else:
		logger.info(f"[AI] LLM call failed, using local fallback")
		transactions = _local_fallback_parse(table_rows, layout_lines, ocr_lines)

	logger.info(f"[AI] Single-call extraction: {len(transactions)} transactions")
	
	return {
		"clean": {
			"metadata": {},
			"transactions": transactions,
		},
		"debug": {
			"num_llm_calls": 1,
			"total_available_chars": total_chars,
			"chunks": 1,
			"mode": "emergency_single_call",
		},
	}
