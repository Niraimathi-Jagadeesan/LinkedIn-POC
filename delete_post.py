"""
delete_post.py — Delete a LinkedIn post by its URN.

Usage:
    python delete_post.py urn:li:ugcPost:1234567890
    python delete_post.py --last          # deletes the most recent dry-run post ID
    python delete_post.py --list          # lists all post IDs saved in dry-run outputs
"""

import sys
import json
from pathlib import Path

from config.settings import get_settings
from linkedin.auth import get_access_token
from linkedin.api_client import LinkedInAPIClient


def list_dry_run_posts() -> list[dict]:
    dry_run_dir = Path("./data/dry_run")
    if not dry_run_dir.exists():
        return []
    posts = []
    for f in sorted(dry_run_dir.glob("*.json"), reverse=True):
        data = json.loads(f.read_text())
        posts.append({"file": f.name, "id": data.get("id"), "format": data.get("format"), "text": data.get("text", "")[:80]})
    return posts


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return

    settings = get_settings()

    if settings.mock_linkedin:
        print("MOCK_LINKEDIN=true — switch to false to delete real posts.")
        return

    if args[0] == "--list":
        posts = list_dry_run_posts()
        if not posts:
            print("No dry-run posts found in data/dry_run/")
            return
        print(f"{'File':<35} {'Format':<10} {'Post ID':<35} Preview")
        print("-" * 110)
        for p in posts:
            print(f"{p['file']:<35} {p['format']:<10} {p['id']:<35} {p['text']}")
        return

    if args[0] == "--last":
        posts = list_dry_run_posts()
        if not posts:
            print("No dry-run post records found.")
            return
        post_urn = posts[0]["id"]
        print(f"Using most recent post ID: {post_urn}")
    else:
        post_urn = args[0]

    print(f"\nDeleting post: {post_urn}")
    confirm = input("Are you sure? This cannot be undone. (yes/no): ").strip().lower()
    if confirm not in ("yes", "y"):
        print("Cancelled.")
        return

    token = get_access_token()
    client = LinkedInAPIClient(token)

    try:
        client.delete_post(post_urn)
        print(f"Post deleted successfully.")
    except Exception as exc:
        print(f"Failed to delete post: {exc}")


if __name__ == "__main__":
    main()
