import numpy as np
import armcore as C
cfg=C.load_config()
q=C.get_pose(cfg,"right","table_approach").copy()
q[1]=-90.6   # shoulder_lift 减小 = 抬高，比 table_hover 高约 5
C.set_pose(cfg,"right","table_approach",q)
C.save_config({"arms":{"right":{"poses":{"table_approach":q.tolist()}}}})
print("right table_approach ->", np.round(q,2).tolist())
