# 0008. An empty ONC listing is a measured zero, and the calendar says so

Status: accepted
Date: 2026-08-27

## Context

`boatphone/onc_client.build_uptime_calendar()` turns an ONC archive listing into a dense,
one-row-per-bin uptime calendar. That calendar is what decides where PlanetScope quota is spent,
and quota is unrecoverable, so every failure mode in it has to resolve to either a loud error or
an honestly-labelled measurement -- never to a plausible-looking table.

One case sits exactly on that line. `EmptyListingError` means **every listing request succeeded
and ONC returned no files for the span**. Two different readings are available:

* "the method found nothing" -- there genuinely is no file in this hour/day. For a short span this
  is a real, expected and useful finding, and it is the one thing the calendar exists to state.
* "the method is broken" -- a whole-study-window query returning nothing is a broken request, not
  an empty ocean.

CLAUDE.md invariant 9 forbids merging those two claims, and decision 0007 already settled the
neighbouring case (two ONC HTTP 400s that are positive not-deployed / not-yet statements).

## Decision

`build_uptime_calendar()` catches `EmptyListingError` **only**, and treats it as a measured zero:
it returns a complete, dense calendar in which every bin is `available = False`.

What keeps the two claims separate:

- **`EmptyListingError` attests the requests SUCCEEDED.** A request that failed raises
  `ONCListingError`, which is *not* caught here and propagates (invariant 5). So the all-`False`
  calendar is only ever produced from a successful, genuinely empty answer.
- **It is printed, every time**, naming the span and how many bins were marked unavailable. There
  is no silent zero.
- `list_fft_files()` still raises on an empty listing, because at its level -- a large span --
  nothing returned is evidence of a broken request rather than of an empty archive.
- Deployment boundaries continue to come from `get_deployments` metadata (D6), never inferred from
  an empty listing. "No file listed" and "not deployed" stay different claims.

## Consequences

- **For a consumer of an all-unavailable calendar:** it means "ONC answered, and listed nothing
  here." It does **not** by itself mean the instrument was down, and it does not mean the request
  failed. Before acting on one, check the deployment metadata: inside a known deployment an
  all-unavailable span is a real outage claim; outside every deployment it is simply not-deployed.
  Either way the Planet-facing action is the same and is the safe one -- **do not spend quota on
  the span** -- because this path withholds orders, it never invents them.
- The failure direction is conservative for quota but not for uptime: a listing hiccup that
  returns an honest-looking empty page would read as an outage and cost us dates we could have
  ordered. That is the cost we accept, and A4's actual pull is where it would surface.
- **How to tell this was wrong:** an all-`False` span inside a deployment that A4 nevertheless
  downloads files for. If that happens, this decision is the first place to look.
