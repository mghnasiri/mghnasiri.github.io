"""
Tim Hortons Hockey Challenge - Player Pool Scraper
===================================================
Scrapes daily eligible players and their group assignments from
timnhlassist.com (SvelteKit __data.json endpoint).

Fallback strategy when scraping fails:
  1. Load cached_pool.json (last successful full scrape)
  2. Fetch today's NHL schedule
  3. Filter cached pool to only players whose teams play today
  4. This preserves correct group assignments even when the source is down

Author: Mohammad G. Nasiri
"""

import requests
import json
import os
import re
import time
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATA_DIR = "data"
    TIMS_DIR = f"{DATA_DIR}/tims_players"
    CACHED_POOL = f"{TIMS_DIR}/cached_pool.json"
    TODAY = datetime.now().strftime("%Y-%m-%d")

    # Primary source: timnhlassist.com (SvelteKit SSR with __data.json)
    PRIMARY_URL = "https://www.timnhlassist.com/__data.json"
    FALLBACK_URL = "https://www.timnhlassist.com/"

    # NHL schedule API (for cache-based fallback)
    NHL_SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule"

    # Request headers to mimic browser
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json, text/html',
    }

    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds between retries

os.makedirs(Config.TIMS_DIR, exist_ok=True)

print("=" * 60)
print("🍩 TIM HORTONS HOCKEY CHALLENGE - PLAYER SCRAPER")
print(f"📅 Date: {Config.TODAY}")
print("=" * 60)


# =============================================================================
# SCRAPING METHODS
# =============================================================================
def fetch_with_retry(url, headers=None, timeout=15):
    """Fetch URL with retry logic. Returns response or None."""
    for attempt in range(1, Config.MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers or Config.HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp
            print(f"   ⚠️ Attempt {attempt}/{Config.MAX_RETRIES}: status {resp.status_code}")
        except requests.exceptions.Timeout:
            print(f"   ⚠️ Attempt {attempt}/{Config.MAX_RETRIES}: timeout after {timeout}s")
        except requests.exceptions.ConnectionError:
            print(f"   ⚠️ Attempt {attempt}/{Config.MAX_RETRIES}: connection error")
        except Exception as e:
            print(f"   ⚠️ Attempt {attempt}/{Config.MAX_RETRIES}: {e}")

        if attempt < Config.MAX_RETRIES:
            delay = Config.RETRY_DELAY * attempt  # progressive backoff
            print(f"   ⏳ Retrying in {delay}s...")
            time.sleep(delay)

    return None


def scrape_from_sveltekit_data():
    """
    Scrape player data from timnhlassist.com's __data.json endpoint.

    SvelteKit uses "devalue" serialization:
      - nodes[1].data is a flat list
      - data[0] = master index with 'data' key pointing to another index
      - data[1] = list of indices where each player record starts
      - data[player_idx] = dict mapping field names to absolute positions
      - data[position] = the actual field value

    Player template fields we care about:
      fullname, player_id, pick_number (group), tri_code (team),
      goals, shots, games, position, last_ten_goals, power_play_goals
    """
    print("\n📡 Method 1: Fetching from __data.json (up to 3 retries)...")

    try:
        resp = fetch_with_retry(Config.PRIMARY_URL)
        if not resp:
            print("   ❌ All retries failed for __data.json")
            return None

        raw_data = resp.json()
        nodes = raw_data.get('nodes', [])

        # Find the data node (usually nodes[1])
        data_node = None
        for node in nodes:
            if isinstance(node, dict) and 'data' in node:
                data_node = node['data']
                break

        if not data_node or not isinstance(data_node, list) or len(data_node) < 3:
            print("   ⚠️ Could not find SvelteKit data node")
            return None

        data = data_node
        all_players_found = []

        # data[0] is master index, data[1] is list of player indices
        if not isinstance(data[1], list):
            print("   ⚠️ Unexpected data format: data[1] is not a list")
            return None

        player_indices = data[1]
        print(f"   Found {len(player_indices)} player entries")

        # Fields we want to extract
        target_fields = {
            'fullname', 'firstname', 'lastname', 'player_id', 'pick_number',
            'tri_code', 'goals', 'shots', 'games', 'position',
            'last_ten_goals', 'power_play_goals', 'home_away',
        }

        for idx in player_indices:
            try:
                if idx >= len(data):
                    continue

                template = data[idx]
                if not isinstance(template, dict):
                    continue

                # Resolve field values from their absolute positions
                player = {}
                for field_name, pos in template.items():
                    if field_name not in target_fields:
                        continue
                    if isinstance(pos, int) and pos < len(data):
                        player[field_name] = data[pos]
                    else:
                        player[field_name] = pos

                # Build normalized player record
                name = player.get('fullname', '')
                if not name:
                    first = player.get('firstname', '')
                    last = player.get('lastname', '')
                    if isinstance(first, str) and isinstance(last, str):
                        name = f"{first} {last}".strip()

                if not isinstance(name, str) or not name:
                    continue

                player_id = player.get('player_id', 0)
                pick_num = player.get('pick_number', 0)
                team = player.get('tri_code', '')
                if isinstance(team, str):
                    team = team.upper()
                else:
                    team = ''

                goals = player.get('goals', 0)
                if not isinstance(goals, (int, float)):
                    goals = 0

                record = {
                    'player_id': player_id,
                    'name': name,
                    'team': team,
                    'position': str(player.get('position', '')),
                    'group': pick_num,  # Tim Hortons pick group
                    'goals': int(goals),
                    'shots': int(player.get('shots', 0)) if isinstance(player.get('shots'), (int, float)) else 0,
                    'games': int(player.get('games', 0)) if isinstance(player.get('games'), (int, float)) else 0,
                    'last_ten_goals': int(player.get('last_ten_goals', 0)) if isinstance(player.get('last_ten_goals'), (int, float)) else 0,
                    'pp_goals': int(player.get('power_play_goals', 0)) if isinstance(player.get('power_play_goals'), (int, float)) else 0,
                }
                all_players_found.append(record)

            except (IndexError, TypeError, KeyError):
                continue

        if not all_players_found:
            print("   ⚠️ Could not extract player data from __data.json")
            return None

        print(f"   ✅ Found {len(all_players_found)} players")
        return all_players_found

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def scrape_from_html():
    """
    Fallback: scrape player data from HTML page content.
    Looks for embedded JSON in script tags.
    """
    print("\n📡 Method 2: Fetching from HTML page (up to 3 retries)...")

    try:
        resp = fetch_with_retry(Config.FALLBACK_URL)
        if not resp:
            print("   ❌ All retries failed for HTML page")
            return None

        html = resp.text

        # Look for SvelteKit embedded data in script tags
        # Pattern: <script>__sveltekit_data = {...}</script>
        patterns = [
            r'__sveltekit_\w+\s*=\s*(\{.+?\})\s*;?\s*</script>',
            r'type="application/json"\s*>(\{.+?\})</script>',
            r'data-sveltekit-[\w-]+="[^"]*"[^>]*>(\{.+?\})</script>',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, html, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    # Try to extract players from this data
                    players = extract_players_from_html_data(data)
                    if players:
                        return players
                except json.JSONDecodeError:
                    continue

        # Also try to extract player names from the visible HTML
        players = extract_players_from_visible_html(html)
        if players:
            return players

        print("   ⚠️ Could not extract player data from HTML")
        return None

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def extract_players_from_html_data(data):
    """Extract player records from parsed HTML JSON data"""
    players = []

    def walk(obj, depth=0):
        if depth > 8:
            return
        if isinstance(obj, dict):
            if ('fullname' in obj or 'player_id' in obj) and ('goals' in obj or 'shots' in obj):
                name = obj.get('fullname', f"{obj.get('firstname', '')} {obj.get('lastname', '')}".strip())
                if name:
                    players.append({
                        'player_id': obj.get('player_id', 0),
                        'name': name,
                        'team': obj.get('tri_code', ''),
                        'position': obj.get('position', ''),
                        'goals': obj.get('goals', 0),
                        'shots': obj.get('shots', 0),
                        'games': obj.get('games', 0),
                    })
            for v in obj.values():
                walk(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, depth + 1)

    walk(data)
    return players if players else None


def extract_players_from_visible_html(html):
    """Extract player names from visible HTML content as last resort"""
    # Look for player name patterns in the HTML
    # This is the least reliable method
    name_pattern = r'(?:Pick\s*#?\s*\d[^<]*?)((?:[A-Z][a-z]+\s)+[A-Z][a-z]+)'
    matches = re.findall(name_pattern, html)
    if matches:
        players = [{'name': m.strip(), 'player_id': 0, 'team': '', 'position': ''} for m in matches]
        return players
    return None


def assign_groups(players, num_groups=3):
    """
    Assign players to groups if group assignment wasn't scraped.
    Uses a round-robin approach based on scraped ordering,
    which typically follows the Tim Hortons group structure.
    """
    groups = {str(i+1): [] for i in range(num_groups)}

    if not players:
        return groups

    # If players already have group assignments, use them
    if any(p.get('group') for p in players):
        for p in players:
            gid = str(p.get('group', 1))
            if gid in groups:
                groups[gid].append(p)
        return groups

    # Otherwise, distribute evenly (sites usually list in group order)
    chunk_size = len(players) // num_groups
    remainder = len(players) % num_groups

    idx = 0
    for g in range(num_groups):
        size = chunk_size + (1 if g < remainder else 0)
        for _ in range(size):
            if idx < len(players):
                groups[str(g + 1)].append(players[idx])
                idx += 1

    return groups


# =============================================================================
# CACHE FALLBACK
# =============================================================================
def get_teams_playing_today():
    """Fetch today's NHL schedule and return set of team abbreviations."""
    try:
        url = f"{Config.NHL_SCHEDULE_URL}/{Config.TODAY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return set()

        schedule = resp.json()
        teams = set()
        for week in schedule.get('gameWeek', []):
            if week.get('date') == Config.TODAY:
                for g in week.get('games', []):
                    if g.get('gameState') in ('FUT', 'PRE', 'LIVE'):
                        teams.add(g['awayTeam']['abbrev'])
                        teams.add(g['homeTeam']['abbrev'])
        return teams
    except Exception as e:
        print(f"   ⚠️ Could not fetch schedule: {e}")
        return set()


def save_cached_pool(all_players, groups):
    """
    Save the FULL master roster of all Tim Hortons players ever seen.
    Merges with existing cache so we accumulate players across days.
    Also saves today's group assignments as a reference.
    """
    existing_roster = {}

    # Load existing cache and merge
    if os.path.exists(Config.CACHED_POOL):
        try:
            with open(Config.CACHED_POOL, 'r', encoding='utf-8') as f:
                old_cache = json.load(f)
            for p in old_cache.get('master_roster', []):
                key = p.get('name', '')
                if key:
                    existing_roster[key] = p
        except Exception:
            pass

    # Merge today's players into master roster (overwrite with fresh data)
    for p in all_players:
        name = p.get('name', '')
        if name:
            existing_roster[name] = {
                'name': name,
                'player_id': p.get('player_id', 0),
                'team': p.get('team', ''),
                'position': p.get('position', ''),
            }

    cache = {
        "last_updated": datetime.now().isoformat(),
        "last_scrape_date": Config.TODAY,
        "master_roster": list(existing_roster.values()),
        "last_groups": groups,
    }
    with open(Config.CACHED_POOL, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"   💾 Updated master roster ({len(existing_roster)} total players)")


def load_cached_pool_for_today():
    """
    Load master roster and filter to only teams playing today.
    Distributes players evenly into 3 groups (since we can't know
    Tim Hortons' actual group assignments when the site is down).
    Returns (players_list, groups_dict) or (None, None) if no cache.
    """
    if not os.path.exists(Config.CACHED_POOL):
        print("   ⚠️ No cached pool found")
        return None, None

    try:
        with open(Config.CACHED_POOL, 'r', encoding='utf-8') as f:
            cache = json.load(f)

        master_roster = cache.get('master_roster', [])
        last_date = cache.get('last_scrape_date', 'unknown')

        if not master_roster:
            return None, None

        # Get today's teams
        teams_today = get_teams_playing_today()
        if not teams_today:
            print("   ⚠️ Could not determine today's teams")
            return None, None

        print(f"   📋 Master roster: {len(master_roster)} players (last scrape: {last_date})")
        print(f"   🏒 Teams playing today: {', '.join(sorted(teams_today))}")

        # Filter to players whose teams play today
        todays_players = [p for p in master_roster if p.get('team', '') in teams_today]

        if not todays_players:
            print("   ⚠️ No roster players match today's teams")
            return None, None

        # Distribute into 3 groups (round-robin by team for balance)
        todays_players.sort(key=lambda p: (p.get('team', ''), p.get('name', '')))
        groups = {'1': [], '2': [], '3': []}
        for i, p in enumerate(todays_players):
            gid = str((i % 3) + 1)
            groups[gid].append(p)

        print(f"   ✅ {len(todays_players)} players from today's {len(teams_today)} teams")
        for gid in sorted(groups.keys()):
            print(f"      📦 Group {gid}: {len(groups[gid])} players")

        return todays_players, groups

    except Exception as e:
        print(f"   ⚠️ Cache read error: {e}")
        return None, None


# =============================================================================
# MAIN
# =============================================================================

# Try scraping methods in order
players = scrape_from_sveltekit_data()
if not players:
    players = scrape_from_html()

# Sanity gate: a silent format change upstream can parse a tiny, partial list
# that would otherwise count as a "successful" scrape AND overwrite the cached
# pool (poisoning the fallback for days). Reject implausibly small scrapes and
# fall through to the cache instead. Even a 1-game slate has ~8 Tims players.
MIN_EXPECTED_PLAYERS = 6
if players and len(players) < MIN_EXPECTED_PLAYERS:
    print(f"\n⚠️ Only {len(players)} player(s) scraped (< {MIN_EXPECTED_PLAYERS}) "
          f"— treating as a partial/broken scrape; falling back to cache.")
    players = None

if players:
    print(f"\n✅ Successfully scraped {len(players)} players")

    # Assign to groups
    groups = assign_groups(players)

    for gid, gplayers in groups.items():
        print(f"   📦 Group {gid}: {len(gplayers)} players")
        for p in gplayers[:3]:
            print(f"      - {p['name']} ({p.get('team', '?')})")
        if len(gplayers) > 3:
            print(f"      ... and {len(gplayers) - 3} more")

    # Update the master roster cache for future fallback
    save_cached_pool(players, {
        gid: [
            {
                'name': p['name'],
                'player_id': p.get('player_id', 0),
                'team': p.get('team', ''),
                'position': p.get('position', ''),
            }
            for p in gplayers
        ]
        for gid, gplayers in groups.items()
    })

    output = {
        "date": Config.TODAY,
        "source": "timnhlassist.com",
        "players_count": len(players),
        "groups": {
            gid: [
                {
                    'name': p['name'],
                    'player_id': p.get('player_id', 0),
                    'team': p.get('team', ''),
                    'position': p.get('position', ''),
                }
                for p in gplayers
            ]
            for gid, gplayers in groups.items()
        },
        "all_players": [
            {
                'name': p['name'],
                'player_id': p.get('player_id', 0),
                'team': p.get('team', ''),
                'position': p.get('position', ''),
                'goals': p.get('goals', 0),
                'shots': p.get('shots', 0),
            }
            for p in players
        ],
        "scraped_at": datetime.now().isoformat()
    }

else:
    # Scraping failed — try cached pool fallback
    print("\n⚠️ Scraping failed, trying cached pool fallback...")
    cached_players, cached_groups = load_cached_pool_for_today()

    if cached_players and cached_groups:
        print(f"\n✅ Using cached pool ({len(cached_players)} players for today)")
        output = {
            "date": Config.TODAY,
            "source": "cached_pool",
            "players_count": len(cached_players),
            "groups": cached_groups,
            "all_players": cached_players,
            "scraped_at": datetime.now().isoformat(),
            "note": "Scraped from cache — source was unavailable"
        }
    else:
        print("\n❌ No cached pool available either")
        print("   The prediction script will use all players from today's games")
        output = {
            "date": Config.TODAY,
            "source": "none",
            "players_count": 0,
            "groups": {},
            "all_players": [],
            "scraped_at": datetime.now().isoformat(),
            "error": "Could not scrape Tim Hortons player pool"
        }

# Save
output_file = f"{Config.TIMS_DIR}/{Config.TODAY}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n💾 Saved: {output_file}")

# Also save as latest
latest_file = f"{Config.TIMS_DIR}/latest.json"
with open(latest_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"💾 Saved: {latest_file}")

print("\n✅ Tim Hortons scraper complete!")
