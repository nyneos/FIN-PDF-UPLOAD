"""
Token usage tracking for Groq API rate limit management.
Tracks daily and monthly token consumption to prevent hitting TPM limits.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from logging_config import configure_logging

logger = configure_logging()

# Storage file for token usage tracking (cross-platform)
TRACKER_FILE = Path(tempfile.gettempdir()) / "bank_parser_token_tracker.json"


def _get_current_period():
	"""Get current day and month identifiers."""
	now = datetime.now()
	return {
		"day": now.strftime("%Y-%m-%d"),
		"month": now.strftime("%Y-%m"),
		"hour": now.strftime("%Y-%m-%d-%H"),
	}


def _load_tracker():
	"""Load token usage data from disk."""
	if not TRACKER_FILE.exists():
		return {"daily": {}, "monthly": {}, "hourly": {}}
	
	try:
		with open(TRACKER_FILE, "r") as f:
			return json.load(f)
	except Exception as e:
		logger.warning(f"[TOKEN_TRACKER] Failed to load tracker: {e}")
		return {"daily": {}, "monthly": {}, "hourly": {}}


def _save_tracker(data):
	"""Save token usage data to disk."""
	try:
		with open(TRACKER_FILE, "w") as f:
			json.dump(data, f, indent=2)
	except Exception as e:
		logger.error(f"[TOKEN_TRACKER] Failed to save tracker: {e}")


def _cleanup_old_data(tracker, current_day, current_month, current_hour):
	"""Remove data older than 31 days to prevent unbounded growth."""
	cutoff = datetime.now() - timedelta(days=31)
	cutoff_day = cutoff.strftime("%Y-%m-%d")
	cutoff_month = cutoff.strftime("%Y-%m")
	cutoff_hour = cutoff.strftime("%Y-%m-%d-%H")
	
	# Clean old daily entries
	tracker["daily"] = {k: v for k, v in tracker["daily"].items() if k >= cutoff_day}
	
	# Clean old monthly entries
	tracker["monthly"] = {k: v for k, v in tracker["monthly"].items() if k >= cutoff_month}
	
	# Clean old hourly entries (keep last 48 hours)
	tracker["hourly"] = {k: v for k, v in tracker["hourly"].items() if k >= cutoff_hour}


def record_tokens(tokens_used: int):
	"""
	Record token usage for the current day and month.
	
	Args:
		tokens_used: Number of tokens consumed in this API call
	"""
	period = _get_current_period()
	tracker = _load_tracker()
	
	# Initialize if missing
	if period["day"] not in tracker["daily"]:
		tracker["daily"][period["day"]] = 0
	if period["month"] not in tracker["monthly"]:
		tracker["monthly"][period["month"]] = 0
	if period["hour"] not in tracker["hourly"]:
		tracker["hourly"][period["hour"]] = 0
	
	# Update counts
	tracker["daily"][period["day"]] += tokens_used
	tracker["monthly"][period["month"]] += tokens_used
	tracker["hourly"][period["hour"]] += tokens_used
	
	# Cleanup old data
	_cleanup_old_data(tracker, period["day"], period["month"], period["hour"])
	
	# Save
	_save_tracker(tracker)
	
	logger.info(f"[TOKEN_TRACKER] Recorded {tokens_used} tokens (day={tracker['daily'][period['day']]}, month={tracker['monthly'][period['month']]})")


def get_usage_stats(tpm_limit: int = 30000, tokens_per_month: int = 14400000):
	"""
	Get current token usage statistics.
	
	Args:
		tpm_limit: Tokens per minute limit (Groq free tier = 30,000 TPM)
		tokens_per_month: Monthly token limit (Groq free tier = 14.4M/month)
	
	Returns:
		dict with usage stats and eligibility status
	"""
	period = _get_current_period()
	tracker = _load_tracker()
	
	# Get current usage
	tokens_today = tracker["daily"].get(period["day"], 0)
	tokens_month = tracker["monthly"].get(period["month"], 0)
	tokens_hour = tracker["hourly"].get(period["hour"], 0)
	
	# Calculate remaining
	# Note: TPM is per-minute, but we track hourly as proxy
	# Assume 60 minutes in hour, so hourly_limit = tpm_limit * 60
	hourly_limit = tpm_limit * 60
	remaining_hour = max(0, hourly_limit - tokens_hour)
	remaining_month = max(0, tokens_per_month - tokens_month)
	
	# Eligibility: can we make a typical chunk call (~3k tokens)?
	typical_chunk_tokens = 3000
	eligible = (remaining_hour >= typical_chunk_tokens) and (remaining_month >= typical_chunk_tokens)
	
	return {
		"current_period": {
			"day": period["day"],
			"month": period["month"],
			"hour": period["hour"],
		},
		"usage": {
			"tokens_today": tokens_today,
			"tokens_this_month": tokens_month,
			"tokens_this_hour": tokens_hour,
		},
		"limits": {
			"tpm_limit": tpm_limit,
			"hourly_limit": hourly_limit,
			"monthly_limit": tokens_per_month,
		},
		"remaining": {
			"hour": remaining_hour,
			"month": remaining_month,
		},
		"eligibility": {
			"can_make_request": eligible,
			"estimated_chunks_available": min(
				remaining_hour // typical_chunk_tokens,
				remaining_month // typical_chunk_tokens,
			),
		},
		"warnings": _generate_warnings(tokens_hour, hourly_limit, tokens_month, tokens_per_month),
	}


def _generate_warnings(tokens_hour, hourly_limit, tokens_month, monthly_limit):
	"""Generate warning messages based on usage levels."""
	warnings = []
	
	hour_pct = (tokens_hour / hourly_limit * 100) if hourly_limit > 0 else 0
	month_pct = (tokens_month / monthly_limit * 100) if monthly_limit > 0 else 0
	
	if hour_pct >= 90:
		warnings.append(f"Hourly usage at {hour_pct:.1f}% - rate limiting likely")
	elif hour_pct >= 70:
		warnings.append(f"Hourly usage at {hour_pct:.1f}% - approaching limit")
	
	if month_pct >= 90:
		warnings.append(f"Monthly usage at {month_pct:.1f}% - quota nearly exhausted")
	elif month_pct >= 70:
		warnings.append(f"Monthly usage at {month_pct:.1f}% - monitor usage closely")
	
	return warnings


def reset_tracker():
	"""Reset all tracking data (admin/testing only)."""
	if TRACKER_FILE.exists():
		TRACKER_FILE.unlink()
	logger.info("[TOKEN_TRACKER] Reset complete")
