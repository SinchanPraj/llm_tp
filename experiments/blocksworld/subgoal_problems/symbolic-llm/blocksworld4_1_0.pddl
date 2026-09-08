
    (define (problem prob-0)
        (:domain blocksworld)
        (:objects b1 b2 b3 b4 - block
    t1 t2 t3 t4 t5 t6 - table)
        (:init 
            (arm-empty)
    (on b1 b2)
    (on-table b2 t4)

    (on b3 b4)
    (on-table b4 t3)

    (clear b1)
    (clear b3)
    (clear-table t6)
    (clear-table t1)
    (clear-table t5)
    (clear-table t2)
        )      
        (:goal (and (on-table b3 t3) (on b4 b3) (on-table b1 t4) (on b2 b1)))
        
    )
    