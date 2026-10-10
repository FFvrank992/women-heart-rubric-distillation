# Blind pass 1 — 100 posts

Label all 100 rows yourself, from scratch. **Do not open
`data/teacher_labels.csv`, and do not ask any chatbot what these posts are.**
These rows are the only measurement in the project that no model has touched,
and that is the whole of their value. The tools will refuse to show you model
output for them until both passes are done.

Read `RUBRIC.md` end to end first, including the boundary cases: that is where
the hard decisions are, and most of them are there because a real post in this
corpus is genuinely ambiguous.

When this sheet is finished, wait **at least three days**, then run
`python pilot.py --pass2` and label the same rows again in the new order,
without reopening this file. The agreement between the two passes is the
reliability number for this annotator.
