"""
AI Verification Layer:
Normalizes table rows into structured transactions using Groq.
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
	"""Attempt to find and parse JSON embedded in LLM text.

	Strategies (in order):
	- direct json.loads(text)
	- extract ```json ... ``` or ``` ... ``` code fence
	- find first '{' or '[' and extract balanced JSON substring
	"""
	# New robust extractor:
	# 1) Remove Qwen <think> blocks
	# 2) Try fenced JSON, direct JSON, or best-effort substring extraction
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
	"""Coerce common AI output shapes into the required transaction schema.

	Accepts rows like {date, amount, type, description} or already-correct rows.
	Returns a new list where each item has keys:
	tran_date, value_date, narration, withdrawal, deposit, balance
	"""
	out = []
	if not rows:
		return out

	for r in rows:
		if not isinstance(r, dict):
			# skip invalid
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

		# Amount extraction: look for explicit deposit/withdrawal keys first
		# then fallback to 'amount' + 'type' mapping
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
			# search common debit/credit keys
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
				# if no type, heuristically leave as deposit (but could be ambiguous)
				item["deposit"] = amt

		# if balance present as string, coerce
		if isinstance(item.get("balance"), str):
			item["balance"] = to_float(item.get("balance"))

		# keep original dates as strings; final normalizer will convert formats
		out.append(item)

	return out


def verify_and_clean(layout_data, table_data, ocr_data):
	layout_lines = flatten_layout(layout_data)
	table_rows = table_data
	ocr_lines = ocr_data or []

	# Build candidate content and ALWAYS truncate aggressively to avoid 413 errors.
	# Aim for ~12KB safe margin (Groq Scout 17B max is ~30k tokens, ~6 chars/token on avg).
	SAFE_PAYLOAD_CHARS = 8000  # well under Groq token limits
	
	joined_layout = "\n".join(layout_lines)
	joined_table = "\n".join(_join_table_rows(table_rows))
	joined_ocr = "\n".join(ocr_lines)

	assembled = "\n\n".join([joined_layout, joined_table, joined_ocr])
	char_len = len(assembled)

	logger.info(f"[AI] Assembled payload chars={char_len}")

	# Always truncate to safe size before sending to LLM
	truncated_info = None
	if char_len > SAFE_PAYLOAD_CHARS:
		logger.info(f"[AI] Payload {char_len} exceeds safe limit {SAFE_PAYLOAD_CHARS}, truncating...")
		truncated = _truncate_payload(layout_lines, table_rows, ocr_lines, SAFE_PAYLOAD_CHARS)
		truncated_info = truncated
		prompt_content = (
			"layout_text:\n"
			+ "\n".join(truncated["layout"])
			+ "\n\n"
			+ "table_data:\n"
			+ "\n".join(truncated["table"])
			+ "\n\n"
			+ "ocr_data:\n"
			+ "\n".join(truncated["ocr"])
		)
	else:
		prompt_content = f"layout_text:\n{joined_layout}\n\ntable_data:\n{joined_table}\n\nocr_data:\n{joined_ocr}"

	prompt_template = """
	You are an expert financial statement parser.

	You will receive 3 sections: layout_text, table_data and ocr_data.

	Your job has TWO responsibilities:

	────────────────────────────────────────
	1) Extract FULL METADATA from the PDF.
	────────────────────────────────────────
	You MUST extract the following fields using patterns found inside layout_text, 
	table_data and ocr_data:

	- account_number: detect 8–20 digit number or masked format (XXXX1234).
	- account_name: customer name printed on statement header or address section.
	- bank_name: detect from known names (Axis Bank, UCO Bank, SBI, HDFC, ICICI, Kotak, Yes Bank, etc.)
	- ifsc: regex [A-Z]{4}[0-9]{7}
	- micr: 9 digit code.
	- period_start, period_end:
	    Extract from lines like:
	      “Statement from DD-MM-YYYY to DD-MM-YYYY”
	      “Between … and …”
	      “Period: …”
	    Normalize dates to YYYY-MM-DD.
	- opening_balance:
	      First visible “Opening Balance” OR earliest table balance.
	- closing_balance:
	      Last visible “Closing Balance” OR last table balance.

	If ANY metadata value is not present, return null (do NOT guess).

	────────────────────────────────────────
	2) Extract ALL transactions and normalize.
	────────────────────────────────────────
	Each transaction must follow schema:
	{
	  "tran_date": "YYYY-MM-DD",
	  "value_date": "YYYY-MM-DD",
	  "narration": "",
	  "withdrawal": number or null,
	  "deposit": number or null,
	  "balance": number or null
	}

	Rules:
	- Accept dates in DD-MM-YYYY, DD/MM/YYYY, YY formats, etc. Normalize to YYYY-MM-DD.
	- If only one date is present → use for both tran_date and value_date.
	- Deduce withdrawal/deposit using DR, CR, -, +, or column position.
	- Remove all non-transaction rows:
	      "Opening Balance", "Closing Balance", "Page", "Summary", "Reward", etc.
	- Narration:
	      must NOT include dates, amounts, column headers, or garbage.
	- All numbers must be proper floats (no commas).
	- If any field is missing → set to null.

	────────────────────────────────────────
	MANDATORY FINAL OUTPUT (ONLY THIS JSON)
	────────────────────────────────────────
	{
	  "metadata": {
	    "account_number": ..., 
	    "account_name": ..., 
	    "bank_name": ..., 
	    "ifsc": ..., 
	    "micr": ..., 
	    "period_start": ..., 
	    "period_end": ..., 
	    "opening_balance": ..., 
	    "closing_balance": ...
	  },
	  "transactions": [...]
	}

	NO EXTRA TEXT. NO EXPLANATION. JSON ONLY.
	If no transactions found → transactions: [].
	If metadata unavailable → return null for those fields.

	<<PROMPT_CONTENT>>

	Return ONLY the JSON object above.
	"""

	# Build prompt safely: avoid f-strings or .format on the template so braces
	# inside the JSON schema are preserved exactly as written.
	prompt = prompt_template.replace('<<PROMPT_CONTENT>>', prompt_content)
	# Add a strict guard instruction
	prompt = prompt + "\n\nRETURN ONLY JSON. NO TEXT. NO THINK TAGS."

	logger.info("[AI] Sending extraction to Groq...")

	# Fast-path: try a single-shot combined metadata+transactions call.
	# If successful, we avoid the separate metadata + transactions + regulator calls
	# which reduces the number of LLM requests (common-case optimization).
	try:
		raw_single = groq_llm([
			{"role": "system", "content": "You are an expert bank statement parser. Return JSON only, no explanation."},
			{"role": "user", "content": prompt},
		])
		# strip any think-mode wrappers
		raw_single = re.sub(r"<think>.*?</think>", "", raw_single, flags=re.DOTALL | re.IGNORECASE)
		raw_single = raw_single.replace("<think>", "").replace("</think>", "")
		try:
			parsed_single = _extract_json_from_text(raw_single)
		except JSONDecodeError:
			parsed_single = None
		if parsed_single is not None:
			# If the model returned useful structured output, return it immediately.
			if isinstance(parsed_single, dict):
				if any(k in parsed_single for k in ("transactions", "opening_balance", "metadata")):
					logger.info("[AI] Single-shot parsed combined metadata+transactions")
					return {"clean": parsed_single, "debug": {"ai_raw": raw_single[:1000]}}
				# If dict contains a list under a common key, return that
				for key in ("result", "data", "transactions", "items"):
					if key in parsed_single and isinstance(parsed_single[key], list):
						logger.info(f"[AI] Single-shot extracted list from key={key}")
						return parsed_single[key]
				# otherwise wrap dict as clean
				return {"clean": parsed_single, "debug": {"ai_raw": raw_single[:1000]}}
			if isinstance(parsed_single, list):
				logger.info(f"[AI] Single-shot parsed {len(parsed_single)} transactions")
				return parsed_single
	except Exception as e:
		logger.warning(f"[AI] Single-shot attempt failed: {e}; falling back to local parse")
		# If LLM fails, use local fallback (NO extra LLM calls)
		local_txns = _local_fallback_parse(table_rows, layout_lines, ocr_lines)
		result = {
			"clean": {
				"metadata": {},
				"opening_balance": None,
				"transactions": local_txns,
				"validation": {},
			},
			"debug": {
				"ai_error": str(e),
				"payload_chars": char_len,
				"fallback_used": "local_parse_on_ai_fail",
			},
		}
		return result

		# Agent 1: metadata extractor (uses layout only)
		try:
			meta_prompt = (
				"You are a metadata extractor. Return only a JSON object with the fields:"
				" account_number, account_name, bank_name, ifsc, micr, period_start, period_end,"
				" opening_balance, closing_balance. Use layout_text only.\n\n"
				+ "layout_text:\n"
				+ "\n".join(layout_lines)
				+ "\n\nRETURN ONLY JSON. NO TEXT. NO THINK TAGS."
			)
			raw_meta = groq_llm([
				{"role": "system", "content": "You extract metadata from bank statements."},
				{"role": "user", "content": meta_prompt},
			])
			# Strip think-mode and parse
			try:
				meta_parsed = _extract_json_from_text(raw_meta)
			except JSONDecodeError:
				meta_parsed = None
		except Exception:
			meta_parsed = None

		# Agent 2: transactions extractor (use truncated payload)
		try:
			truncated = _truncate_payload(layout_lines, table_rows, ocr_lines, config.AI_MAX_PAYLOAD_CHARS)
			# Stronger transaction prompt: require the exact schema and provide examples
			tx_prompt = (
				"You are a transaction extractor. Return ONLY JSON with key 'transactions' which is a list of transaction objects.\n\n"
				"REQUIRED transaction schema for each item (fields MUST be present, but may be null):\n"
				"{\n  \"tran_date\": \"YYYY-MM-DD\",\n  \"value_date\": \"YYYY-MM-DD\",\n  \"narration\": \"string\",\n  \"withdrawal\": number or null,\n  \"deposit\": number or null,\n  \"balance\": number or null\n}\n\n"
				"Rules:\n"
				"- Normalize dates to YYYY-MM-DD. If only one date present, use it for both tran_date and value_date.\n"
				"- If the original row shows Debit/DR or amount in Debit column, populate 'withdrawal' with a positive number and set 'deposit' to null.\n"
				"- If the original row shows Credit/CR or amount in Credit column, populate 'deposit' and set 'withdrawal' to null.\n"
				"- If a row has both an amount and a running balance, set 'balance' to that number.\n"
				"- All numeric values must be plain numbers (floats), no commas or currency symbols.\n"
				"- If you cannot infer a specific field, set it to null (do NOT fabricate).\n\n"
				"Example input -> output mapping (these are required formats):\n"
				"Input table row: ['13-11-2025', '', 'UPI/xxx/Some Merchant', '60.00', '', '58.90']\n"
				"Output transaction: {\"tran_date\":\"2025-11-13\", \"value_date\":\"2025-11-13\", \"narration\":\"UPI/xxx/Some Merchant\", \"withdrawal\":60.0, \"deposit\":null, \"balance\":58.9 }\n\n"
				"If the model's natural output would use alternate keys like 'date'/'amount'/'type', map them to the required schema in the returned JSON. Always return the required schema.\n\n"
				+ "table_data:\n"
				+ "\n".join(truncated.get("table", []))
				+ "\n\nlayout_text:\n"
				+ "\n".join(truncated.get("layout", []))
				+ "\n\nRETURN ONLY JSON. NO TEXT. NO THINK TAGS."
			)
			raw_tx = groq_llm([
				{"role": "system", "content": "You extract transactions from bank statements."},
				{"role": "user", "content": tx_prompt},
			])
			try:
				tx_parsed = _extract_json_from_text(raw_tx)
			except JSONDecodeError:
				tx_parsed = None
		except Exception:
			tx_parsed = None

		# Merge agent outputs if available, otherwise fallback to the single full prompt call
		if meta_parsed or tx_parsed:
			combined = {"metadata": meta_parsed or {}, "transactions": []}
			if tx_parsed:
				# tx_parsed may be {'transactions': [...]} or a list
				tx_list = []
				if isinstance(tx_parsed, dict) and "transactions" in tx_parsed and isinstance(tx_parsed["transactions"], list):
					tx_list = tx_parsed["transactions"]
				elif isinstance(tx_parsed, list):
					tx_list = tx_parsed
				else:
					# try to find list under common keys
					for k in ("result", "data", "items", "transactions"):
						if isinstance(tx_parsed, dict) and k in tx_parsed and isinstance(tx_parsed[k], list):
							tx_list = tx_parsed[k]
							break
				# Coerce alternate key shapes into the required schema
				if tx_list:
					combined["transactions"] = _coerce_transaction_schema(tx_list)

			# Run regulator/master agent to validate, dedupe and improve transactions
			try:
				regulator_prompt = (
					"You are a regulator agent. Given the metadata and collected transactions,"
					" validate running balances, deduplicate near-duplicates, normalize dates and amounts,"
					" and return a JSON object: {\n  \"transactions\": [...],\n  \"prompt_fix\": {\"suggestion\": string, \"reason\": string} \n}\n\n"
					+ "metadata:\n" + json.dumps(combined.get("metadata", {})) + "\n\n"
					+ "transactions:\n" + json.dumps(combined.get("transactions", []))
				)
				raw_reg = groq_llm([
					{"role": "system", "content": "You validate and improve parsed transactions."},
					{"role": "user", "content": regulator_prompt},
				])
				try:
					reg_parsed = _extract_json_from_text(raw_reg)
				except JSONDecodeError:
					reg_parsed = None
				if isinstance(reg_parsed, dict) and "transactions" in reg_parsed:
					combined["transactions"] = reg_parsed.get("transactions")
					# store any prompt improvement suggestions
					if isinstance(reg_parsed.get("prompt_fix"), dict):
						try:
							add_suggestion(reg_parsed.get("prompt_fix"))
						except Exception:
							pass
			except Exception:
				# regulator failure is non-fatal; continue with combined
				logger.warning("[AI] Regulator agent failed or unavailable")

			logger.info("[AI] Using multi-agent outputs (metadata/transactions)")
			# If transactions are empty but layout_lines contain transaction-like rows,
			# attempt a local fallback parse (useful when pdfplumber didn't detect tables).
			if not combined.get("transactions"):
				try:
					local_attempt = _local_fallback_parse(table_rows or layout_lines, layout_lines, ocr_lines)
					if local_attempt:
						combined["transactions"] = local_attempt
						debug_info = {"meta_raw": (raw_meta[:800] if 'raw_meta' in locals() else None), "tx_raw": (raw_tx[:800] if 'raw_tx' in locals() else None), "reg_raw": (raw_reg[:800] if 'raw_reg' in locals() else None), "fallback_used": "local_fallback_on_layout"}
						return {"clean": combined, "debug": debug_info}
				except Exception:
					pass

			return {"clean": combined, "debug": {"meta_raw": (raw_meta[:800] if 'raw_meta' in locals() else None), "tx_raw": (raw_tx[:800] if 'raw_tx' in locals() else None), "reg_raw": (raw_reg[:800] if 'raw_reg' in locals() else None)}}

		# If multi-agent failed, fall back to single-shot full prompt
		raw = groq_llm([
			{"role": "system", "content": "You are an expert bank statement parser. Return JSON only, no explanation."},
			{"role": "user", "content": prompt},
		])

		# Remove Qwen <think> if present immediately
		raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
		raw = raw.replace("<think>", "").replace("</think>", "")

		# Try to extract JSON robustly from raw text
		try:
			parsed = _extract_json_from_text(raw)
		except JSONDecodeError as e:
			logger.error(f"[AI] Failed to parse JSON from AI output: {e}; raw_preview={raw[:800]}")
			parsed = None

		if parsed is None:
			raise JSONDecodeError("No JSON from AI", raw, 0)
		# If model returned a plain list of transactions
		if isinstance(parsed, list):
			logger.info(f"[AI] Parsed {len(parsed)} transactions")
			return parsed
		# If model returned a structured dict (metadata, transactions, opening_balance)
		if isinstance(parsed, dict):
			if any(k in parsed for k in ("transactions", "opening_balance", "metadata")):
				logger.info("[AI] Received structured dict from model")
				return {"clean": parsed, "debug": {"ai_raw": raw[:1000]}}
			# If dict contains a list under a common key, return that
			for key in ("result", "data", "transactions", "items"):
				if key in parsed and isinstance(parsed[key], list):
					logger.info(f"[AI] Extracted list from key={key}")
					return parsed[key]
			# otherwise wrap dict as clean
			return {"clean": parsed, "debug": {"ai_raw": raw[:1000]}}

	except Exception as e:
		logger.error(f"[AI] Groq call failed or no-JSON produced: {e}")
		# Fallback: local parse and return results with debug info
		local_txns = _local_fallback_parse(table_rows, layout_lines, ocr_lines)
		result = {
			"clean": {
				"metadata": {},
				"opening_balance": None,
				"transactions": local_txns,
				"validation": {},
			},
			"debug": {
				"ai_error": str(e),
				"payload_chars": char_len,
				"truncated": bool(truncated_info),
				"truncated_info": truncated_info,
			},
		}
		return result