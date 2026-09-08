
    (define (problem prob-1)
        (:domain gripper)
        (:objects robot1 robot2 robot3 robot4 - robot
    rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 - gripper
    room1 room2 room3 room4 - room
    ball1 ball2 ball3 ball4 - object)
        (:init 
            (at ball1 room1)
(at ball2 room2)
(at ball3 room1)
(at ball4 room2)
(at-robby robot1 room1)
(at-robby robot2 room1)
(at-robby robot3 room3)
(at-robby robot4 room3)
(free robot1 lgripper1)
(free robot1 rgripper1)
(free robot2 lgripper2)
(free robot2 rgripper2)
(free robot3 lgripper3)
(free robot3 rgripper3)
(free robot4 lgripper4)
(free robot4 rgripper4)
        )      
        (:goal (and (at ball2 room3)))
        
    )
    