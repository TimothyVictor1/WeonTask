# Human judgment check (10 minutes, 2 raters)

Purpose: confirm the automatic metrics agree with human eyes, as the brief asks.

1. Pick the two arms being compared, e.g. plain vs crop.
2. For steps 1 to 5, open the pair of images (plain step_k next to crop
   step_k) with the original step_00 visible above them.
3. Rater question, answered blind (do not tell the rater which arm is which,
   swap left/right randomly per pair): "Compared to the original, which image
   damaged the UNCHANGED parts more: left or right? Or equal?"
4. Record answers in a small table: step, rater 1, rater 2.
5. Raters: you plus one other person. 10 pairs total is enough.
6. Report: how often humans picked the same winner as the metrics, and any
   pair where they disagreed (those disagreements are findings, not failures).
