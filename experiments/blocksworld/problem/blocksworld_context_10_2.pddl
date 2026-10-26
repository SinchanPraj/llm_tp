(define (problem BW-rand-1)
  (:domain blocksworld)
  (:objects
    b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 - block ; blocks
    t1 t2 t3 t4 t5 t6 - table ; tables
  )

  (:init
    (arm-empty)
    (on b1 b2)
    (on b2 b3)
    (on-table b3 t1)
    (on b4 b5)
    (on b5 b6)
    (on-table b6 t2)
    (on b7 b8)
    (on b8 b9)
    (on b9 b10)
    (on-table b10 t3)
    (clear b1)
    (clear b4)
    (clear b7)
    (clear-table t4)
    (clear-table t5)
    (clear-table t6)
  )

  (:goal
    (and
    ; first subgoal: change the order of blocks on t3
     (on-table b8 t3)
     (on b10 b8)
     (on b9 b10)
     (on b7 b9)
    ; second subgoal: change the order of blocks on t2
     (on-table b4 t2)
     (on b6 b4)
     (on b5 b6)
    ; third subgoal: change the order of blocks on t1
     (on-table b3 t1)
     (on b1 b3)
     (on b2 b1)
  )
  )
)


