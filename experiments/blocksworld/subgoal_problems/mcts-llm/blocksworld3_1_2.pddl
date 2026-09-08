
    (define (problem prob-2)
        (:domain blocksworld)
        (:objects b1 b2 b3 - block
    t1 t2 t3 t4 t5 t6 - table)
        (:init 
            (arm-empty )
(clear b1)
(clear b2)
(clear b3)
(clear-table t4)
(clear-table t5)
(clear-table t6)
(on-table b1 t3)
(on-table b2 t1)
(on-table b3 t2)
        )      
        (:goal (and (on b3 b1)))
        
    )
    