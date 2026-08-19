uint8 PEDESTRIAN=0
uint8 CYCLIST=1
uint8 CAR=2

# pedestrian, cyclist, car
uint8 label

# The confidence score of the detected object in the range [0-1]
float32 confidence_score

# Track ID assigned by the tracker (-1 if untracked)
int32 track_id

# A 3D bounding box
BoundingBox3d bounding_box