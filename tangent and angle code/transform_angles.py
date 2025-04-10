def transform_angles(angles):
    # Convert angles from [0, 180] to [-90, 90]
    return (angles + 90) % 180 - 90
