def closest_compass_direction(angle):
    # Define the 16 compass points for a -90 to 90 degree range
    compass_directions = [
        "W", "WbN", "WNW", "NWbW", "NW", "NWbN", "NNW", "NbW", 
        "N", "NbE", "NNE", "NEbN", "NE", "NEbE", "ENE", "EbN",
    ]
    
    # Normalize the angle to be between -90 and 90
    # This assumes angle is already in the range of -90 to 90.
    
    # Map the angle to the appropriate index (dividing by 11.25 to get the 16 directions)
    direction_index = round((angle + 90) / 11.25) % 16  # Adjust to the range of 16 directions
    
    return compass_directions[direction_index]
