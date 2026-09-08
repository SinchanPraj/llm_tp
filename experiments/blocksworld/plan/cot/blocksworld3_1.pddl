```python
# Define the problem and initial state
problem = """
(define (problem prob)
  (:domain blocksworld)
  (:objects
    b1 b2 b3 - block
    t1 t2 t3 t4 t5 t6 - table
  )
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
  )
  (:goal
    (and
      (on-table b1 t3)
      (on b3 b1)
      (on b2 b3)
    ))
)
"""

# Define the plan
plan = """
(define (problem prob)
  (:domain blocksworld)
  (:objects
    b1 b2 b3 - block
    t1 t2 t3 t4 t5 t6 - table
  )
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
  )
  (:goal
    (and
      (on-table b1 t3)
      (on b3 b1)
      (on b2 b3)
    ))
)

(define (problem prob)
  (:domain blocksworld)
  (:objects
    b1 b2 b3 - block
    t1 t2 t3 t4 t5 t6 - table
  )
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
  )
  (:goal
    (and
      (on-table b1 t3)
      (on b3 b1)
      (on b2 b3)
    ))
)

(define (problem prob)
  (:domain blocksworld)
  (:objects
    b1 b2 b3 - block
    t1 t2 t3 t4 t5 t6 - table
  )
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
  )
  (:goal
    (and
      (on-table b1 t3)
      (on b3 b1)
      (on b2 b3)
    ))
)
"""

# Print the problem and plan
print(problem)
print(plan)
```