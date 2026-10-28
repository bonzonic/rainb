# Work notes — Lumen Health

## Projects

- Our patient-intake redesign is code-named Project Kestrel. Launch target: 18 November.
- The analytics dashboard vendor we picked is Metabase, self-hosted.
- The design system is called Lantern; tokens live in the lantern-tokens repo.
- Our on-call tool is PagerDuty; the rotation changes every Monday at 9am.
- The usability study for Kestrel had 12 participants, run by Sofia Marquez.

## Decisions

- We chose PostgreSQL over MongoDB for the intake service because forms are relational.
- We dropped the chatbot from Kestrel after the study: 9 of 12 participants ignored it.
- Accessibility target is WCAG 2.2 AA, audited before every release.

## Numbers

- Kestrel cut median intake time from 11 minutes to 4 minutes in the pilot.
- The clinic pilot ran at Pantai Hospital, Bangsar, for six weeks.
