#!/usr/bin/env python3
"""Delete all NiFi process groups on startup for clean slate."""

import os
import sys
import time

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests  # noqa: E402 (must disable warnings before importing requests)

NIFI_ENDPOINT = os.environ.get("NIFI_ENDPOINT", "")
NIFI_USERNAME = os.environ.get("NIFI_USERNAME", "")
NIFI_PASSWORD = os.environ.get("NIFI_PASSWORD", "")
NIFI_SSL_VERIFY = os.environ.get("NIFI_SSL_VERIFY", "false").lower() == "true"


def get_token(session):
    """Authenticate with NiFi and get JWT token."""
    r = session.post(
        f"{NIFI_ENDPOINT}/access/token",
        data={"username": NIFI_USERNAME, "password": NIFI_PASSWORD},
        timeout=10,
    )
    r.raise_for_status()
    return r.text.strip()


def delete_all_process_groups():
    if not NIFI_ENDPOINT:
        print("⏭️  NIFI_ENDPOINT not set, skipping NiFi cleanup")
        return

    print(f"🔌 Connecting to NiFi at {NIFI_ENDPOINT}...")
    session = requests.Session()
    session.verify = NIFI_SSL_VERIFY

    token = None
    for attempt in range(1, 61):
        try:
            token = get_token(session)
            session.headers["Authorization"] = f"Bearer {token}"
            print("✅ Authenticated with NiFi")
            break
        except requests.RequestException:
            print(f"   Waiting for NiFi... ({attempt}/60)")
            time.sleep(2)

    if token is None:
        print("❌ NiFi not ready after 120s, giving up")
        sys.exit(1)

    r = session.get(f"{NIFI_ENDPOINT}/flow/process-groups/root", timeout=10)
    r.raise_for_status()
    flow = r.json()
    process_groups = flow.get("processGroups", [])

    if not process_groups:
        print("   ✓ No process groups to delete")
        return

    for pg in process_groups:
        pg_id = pg["id"]
        pg_name = pg.get("component", {}).get("name", "unknown")
        print(f"   🗑️  Deleting process group: {pg_name} ({pg_id})")

        session.put(
            f"{NIFI_ENDPOINT}/process-groups/{pg_id}",
            json={"id": pg_id, "state": "STOPPED"},
            timeout=30,
        )
        time.sleep(2)

        r = session.delete(f"{NIFI_ENDPOINT}/process-groups/{pg_id}", timeout=30)
        if r.status_code == 200:
            print(f"   ✅ Deleted {pg_name}")
        else:
            print(f"   ⚠️  Failed to delete {pg_name}: {r.status_code}")

    print(f"✅ Deleted {len(process_groups)} process group(s)")


if __name__ == "__main__":
    delete_all_process_groups()
