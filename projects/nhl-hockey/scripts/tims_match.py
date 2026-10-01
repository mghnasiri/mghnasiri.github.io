"""
Match model players to the day's Tim Hortons pool by NHL player_id.

Every Tims entry carries the NHL player_id, but six scripts matched by
normalized name, so players whose Tims name differs from their NHL name
(nicknames, accents) silently dropped out of their group's ranking:
Marner on 12 of 12 days, Raty and Kerfoot 10 of 10, Slafkovsky 4 of 4.
Name matching remains only as a fallback for entries without an id.
"""

import unicodedata


def normalize_name(name):
    """Lowercase, accents stripped (Slafkovský -> slafkovsky), punctuation and
    extra spaces removed."""
    name = unicodedata.normalize('NFKD', name or '').encode('ascii', 'ignore').decode()
    return ' '.join(name.lower().replace('.', '').replace("'", '').replace('-', ' ').split())


def match_tims(players, tims_data, label=''):
    """Return (players in today's pool, each tagged with tims_group;
    {player_id: group}). Incoming order is kept; callers rank afterwards.
    Warns about pool players that have no prediction instead of dropping
    them silently."""
    by_id, by_name, pool = {}, {}, {}
    for gid, entries in (tims_data or {}).get('groups', {}).items():
        for e in entries:
            if isinstance(e, dict) and e.get('player_id'):
                by_id[int(e['player_id'])] = gid
                pool[int(e['player_id'])] = e.get('name', '?')
            else:
                name = e if isinstance(e, str) else (e or {}).get('name', '')
                by_name[normalize_name(name)] = gid

    matched, groups = [], {}
    for p in players:
        pid = p.get('player_id')
        gid = by_id.get(int(pid)) if pid else None
        if gid is None:
            gid = by_name.get(normalize_name(p.get('name', '')))
        if gid is None:
            continue
        p['tims_group'] = gid
        if pid:
            groups[pid] = gid
        matched.append(p)

    missing = [name for pid, name in pool.items() if pid not in groups]
    if missing:
        tag = f' [{label}]' if label else ''
        print(f"  WARNING{tag}: {len(missing)} Tims player(s) have no prediction: "
              f"{', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
    return matched, groups
