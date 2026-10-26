(define (problem prob)
(:domain blocksworld)
(:objects
(:init
(arm-empty)
(on b1 b2)
(on b2 b3)
(on-table b3 t3)
(clear b1)
(clear-table t5)
(clear-table t2)
(clear-table t6)
(clear-table t4)
(clear-table t1)
(:goal
(and
(on-table b1 t3)
(on b3 b1)
(on b2 b3)