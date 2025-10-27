import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def IDM(v, s, delta_v, v0=16.0, T=1.2, a=1.0, b=1.5, s0=2.0, delta=4):
    """
    Compute IDM acceleration for a vehicle.

    Parameters:
    - v : float - current speed of the vehicle (m/s)
    - s : float - gap to the front vehicle (m)
    - delta_v : float - relative speed to front vehicle (v - v_front) (m/s)
    - v0 : float - desired speed (m/s)
    - T : float - desired time headway (s)
    - a : float - maximum acceleration (m/s²)
    - b : float - comfortable deceleration (m/s²)
    - s0 : float - minimum gap (m)
    - delta : int - acceleration exponent (default is 4)

    Returns:
    - acceleration : float - longitudinal acceleration (m/s²)
    """

    # Desired dynamic gap
    s_star = s0 + abs(v * T + (v * delta_v) / (2 * (a * b)**0.5))

    # Prevent division by zero
    if s <= 0:
        s = 0.1

    # IDM acceleration formula
    acc = a * (1 - (v / v0)**delta - (s_star / s)**2)

    return round(acc, 2)

def IDM_with_signal(v, s, delta_v, 
                    v0=16.0, T=1.2, a=1.0, b=1.5, s0=2.0, delta=4, leading=True,
                    signal_state=None, signal_distance=None, reaction_distance=50.0):
    """
    IDM model with traffic signal consideration.

    Parameters:
    - v : float - current speed of the vehicle (m/s)
    - s : float - gap to front vehicle (m)
    - delta_v : float - relative speed to front vehicle (v - v_front) (m/s)
    - v0, T, a, b, s0, delta : standard IDM parameters
    - signal_state : str or None - 'red', 'yellow', or 'green'
    - signal_distance : float or None - distance to signal (m)
    - reaction_distance : float - distance within which to react to red/yellow signal

    Returns:
    - acceleration : float - longitudinal acceleration (m/s²)
    """

    # Desired dynamic gap
    s_star = s0 + abs(v * T + (v * delta_v) / (2 * (a * b)**0.5))
    
    # Prevent division by zero
    if s <= 0:
        s = 0.1

    # Base IDM acceleration
    if leading == True:
        acc_idm = a * (1 - (v / v0)**delta - (s_star / s)**2)
    else:
        acc_idm = a * (1 - (v / v0)**delta)

    if v < 1:
        desired_stop_acc = -3

    # Signal logic: if red or yellow and within reaction zone
    acc_signal = 0.0
    if signal_state in ['red'] and signal_distance is not None:
        if 0 < signal_distance < reaction_distance:
            # Decelerate to stop at the signal using comfortable deceleration
            desired_stop_acc = - (v**2) / (2 * max(signal_distance, 0.1))
            if v < 3:
                desired_stop_acc = - 2 * (v**2) / (2 * max(signal_distance, 0.1))
            if v < 1:
                desired_stop_acc = -3

            acc_signal = max(desired_stop_acc, -3)

    # Final acceleration is minimum of IDM and signal braking to ensure safety
    acc = min(acc_idm, acc_signal) if acc_signal < 0 else acc_idm

    return round(acc, 2)


def process_data_XY(file_name='./sorted_data_final.csv', debug=False):
    """
    This code is for process the data has the same lateral postion for easy to observe
    - Direction 4: sort Local_Y ↓ 
    - Direction 2: sort Local_Y ↑ 
    - Direction 3: sort Local_X ← 
    - Direction 1: sort Local_X → 

    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)  # Replace with your file path

    Sdf = df[df['Direction'] == 4]
    Ndf = df[df['Direction'] == 2]
    Wdf = df[df['Direction'] == 3]
    Edf = df[df['Direction'] == 1]

    avg_local_xS = Sdf.groupby('Lane_ID')['Local_X'].mean().reset_index(name='Avg_Local_X')
    avg_local_xN = Ndf.groupby('Lane_ID')['Local_X'].mean().reset_index(name='Avg_Local_X')
    avg_local_yW = Wdf.groupby('Lane_ID')['Local_Y'].mean().reset_index(name='Avg_Local_X')
    avg_local_yE = Edf.groupby('Lane_ID')['Local_Y'].mean().reset_index(name='Avg_Local_X')

    print(avg_local_xS)
    print(avg_local_xN)
    print(avg_local_yW)
    print(avg_local_yE)

    # vehiclei = df[df['Vehicle_ID'] == 1098]
    # x = vehiclei['Local_X']
    # y = vehiclei['Local_Y']
    # xyplot(x,y)

    # Direction 4: sort Local_Y ↓ 
    df.loc[(df['Direction'] == 4) & (df['Lane_ID'] == 2), 'Local_X'] = -7.5
    df.loc[(df['Direction'] == 4) & (df['Lane_ID'] == 1), 'Local_X'] = -4.5
    df.loc[(df['Direction'] == 4) & (df['Lane_ID'] == 11), 'Local_X'] = -1.5
    

    # Direction 2: sort Local_Y ↑
    df.loc[(df['Direction'] == 2) & (df['Lane_ID'] == 2), 'Local_X'] = 7.5
    df.loc[(df['Direction'] == 2) & (df['Lane_ID'] == 1), 'Local_X'] = 4.5
    df.loc[(df['Direction'] == 2) & (df['Lane_ID'] == 11), 'Local_X'] = 1.5

    sorted_df = df
    # Save the sorted data
    sorted_df.to_csv('sorted_data_final.csv', index=False)



def process_data_direction(file_name='./sorted_data_vehilce.csv', debug=False):
    """
    This code is for sort the data based on time,  lane, direction, and movement
    - Direction 4: sort Local_Y ↓ 
    - Direction 2: sort Local_Y ↑ 
    - Direction 3: sort Local_X ← 
    - Direction 1: sort Local_X → 
    """
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)  # Replace with your file path

    # Filter and sort by Direction logic
    df_d1 = df[df['Direction'] == 4].sort_values(by=['Frame_ID', 'Lane_ID', 'Local_Y'], ascending=[True, True, True])
    df_d2 = df[df['Direction'] == 2].sort_values(by=['Frame_ID', 'Lane_ID', 'Local_Y'], ascending=[True, True, False])
    df_d3 = df[df['Direction'] == 3].sort_values(by=['Frame_ID', 'Lane_ID', 'Local_X'], ascending=[True, True, True])
    df_d4 = df[df['Direction'] == 1].sort_values(by=['Frame_ID', 'Lane_ID', 'Local_X'], ascending=[True, True, False])

    # Combine the sorted pieces
    sorted_df = pd.concat([df_d1, df_d2, df_d3, df_d4], ignore_index=True)

    # Optional: final sort by Frame_ID and Lane_ID for stable global structure
    sorted_df = sorted_df.sort_values(by=['Frame_ID', 'Lane_ID', 'Movement']).reset_index(drop=True)

    # Save the sorted data
    sorted_df.to_csv('sorted_data_direction.csv', index=False)

def process_data_4vehilce(file_name='./sorted_data.csv', debug=False):
    """
    This code is for sort the data based on vehicle within the insection range,
    romving all invalid trajectories

    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)  # Replace with your file path

    # Drop rows with missing critical data
    df = df.dropna(subset=['Local_X', 'Local_Y', 'Lane_ID', 'Frame_ID'])

    # Optional: final sort by Frame_ID and Lane_ID for stable global structure
    sorted_df = df.sort_values(by=['Vehicle_ID','Frame_ID']).reset_index(drop=True)

    # Keep only the specified columns
    columns_to_keep = [
        'Frame_ID','Vehicle_ID','Lane_ID', 'Local_X', 'Local_Y', 
        'v_Vel','v_Acc', 'Movement', 'Direction'
    ]
    sorted_df = sorted_df[columns_to_keep].dropna()

    # remove out of boundary data
    sorted_df =  sorted_df[sorted_df['Lane_ID'] != 0]
    sorted_df =  sorted_df[sorted_df['Lane_ID'] != 9999]

    # remove once happend direction or lane ID
    vehicle_id = sorted_df['Vehicle_ID'].unique()
    for i in vehicle_id:
        vehiclei = sorted_df[sorted_df['Vehicle_ID']== i]
        v_lane = vehiclei['Lane_ID'].unique()
        v_dir = vehiclei['Direction'].unique()
        if len(v_lane) > 1 or len(v_dir) > 1:
            print(i, v_lane, v_dir)
            lane_counts1 = vehiclei['Lane_ID'].value_counts()
            lane_counts2 = vehiclei['Direction'].value_counts()
            lanes_to_remove1 = lane_counts1[lane_counts1 == 1].index.tolist()
            lanes_to_remove2 = lane_counts2[lane_counts2 == 1].index.tolist()
            
            if len(lanes_to_remove1) > 0 and len(lanes_to_remove2) > 0:
                index1 = vehiclei[vehiclei['Lane_ID'].isin(lanes_to_remove1)].index.tolist()
                index2 = vehiclei[vehiclei['Direction'].isin(lanes_to_remove2)].index.tolist()
                index = list(set(index1 + index2)) 
            elif len(lanes_to_remove1) > 0:
                index = vehiclei[vehiclei['Lane_ID'].isin(lanes_to_remove1)].index.tolist()
            elif len(lanes_to_remove2) > 0:
                index = vehiclei[vehiclei['Direction'].isin(lanes_to_remove2)].index.tolist()
            else:
                index = []
            sorted_df = sorted_df.drop(index)

    # Save the sorted data
    sorted_df.to_csv('sorted_data_vehilce.csv', index=False)


def signal_analyze(df, frame_max = 6200, frame_min = 0):
    """
    This code is for analyze the traffic signal timing by plotting speed and time

    """
    
    newdf = df[(df['Frame_ID'] > frame_min) & (df['Frame_ID'] < frame_max) &(df['Direction'] == 4) & ((df['Lane_ID'] == 1) | (df['Lane_ID'] == 2))]
    # Get unique frames
    frames = df['Frame_ID'].unique()
    # Group by Vehicle_ID
    grouped = newdf.groupby('Vehicle_ID')
    # Create figure
    plt.figure(figsize=(10, 6))
    # Plot speed profile for each vehicle
    for vehicle_id, group in grouped:
        plt.plot(group['Frame_ID'], group['Local_Y'])
    # Labeling
    plt.xlabel('Frame ID')
    plt.ylabel('Position (m)')
    plt.title('Space vs Frame ID for Each Vehicle')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Red start and end (2600, 3200),

def get_signal_state_NS(frame_id):
    cycle_length = 1000
    phase = frame_id % cycle_length

    if 200 <= phase < 600:
        return 'green'
    elif 600 <= phase < 630:
        return 'yellow'
    else:
        return 'red'

def get_signal_state_NSL(frame_id):
    cycle_length = 1000
    phase = frame_id % cycle_length

    if 90 <= phase < 160:
        return 'green'
    elif 160 <= phase < 190:
        return 'yellow'
    else:
        return 'red'

def get_signal_state_EW(frame_id):
    cycle_length = 1000
    phase = frame_id % cycle_length

    if 50 <= phase < 80:
        return 'yellow'
    elif 750 <= phase < 1000 or 0 <= phase < 50 :
        return 'green'
    else:
        return 'red'

def get_signal_state_EWL(frame_id):
    cycle_length = 1000
    phase = frame_id % cycle_length

    if 640 <= phase < 710:
        return 'green'
    elif 710 <= phase < 740:
        return 'yellow'
    else:
        return 'red'

## Add the signal timing to data
def process_data_signal(file_name='./sorted_data_final.csv', out_filename='sorted_data_final_selectwST.csv', debug=False):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    signal_analyze(df)
    # Add signal timing
    
    df['Signal_State_NS'] = df['Frame_ID'].apply(get_signal_state_NS)
    df['Signal_State_EW'] = df['Frame_ID'].apply(get_signal_state_EW)
    df['Signal_State_NSL'] = df['Frame_ID'].apply(get_signal_state_NSL)
    df['Signal_State_EWL'] = df['Frame_ID'].apply(get_signal_state_EWL)
    # Save the sorted data
    df.to_csv(out_filename, index=False)

def generate_vehicle(arrival_rate=0.3, num_vehicles=300):
    """
    This code is for generate vehicle arrival times based on arrival rate 
generate_vehicle
    """
    
    # Generate inter-arrival times (exponential distribution)
    inter_arrival_times = np.random.exponential(scale=1/arrival_rate, size=num_vehicles)

    # Compute cumulative arrival times
    arrival_times = np.cumsum(inter_arrival_times)

    # Optional: convert to seconds or frame numbers
    arrival_times = arrival_times.round(1)

    return arrival_times

  
def load_data(file_name='./sorted_data_new.csv'):
    df = pd.read_csv(file_name)  # Replace with your file path
    return df

def xy_plotting(x, y):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='blue')
    plt.title("Vehicle XY Position Plot")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.axis('equal')  # Optional: keep the aspect ratio equal
    plt.show()

def plotting_signal(time, signal_list):
    # Map signal states to numeric values
    state_map = {'red': 0, 'yellow': 1, 'green': 2}
    signal_numeric = [state_map[state] for state in signal_list]

    # Plot
    plt.figure(figsize=(10, 2))
    plt.step(time, signal_numeric, where='post')
    plt.yticks([0, 1, 2], ['Red', 'Yellow', 'Green'])
    plt.xlabel('Time (s)')
    plt.ylabel('Signal State')
    plt.title('Traffic Signal Timeline')
    plt.grid(True)
    plt.tight_layout()
    

def label_colar(signal_state):
        if signal_state == 'red':
            signal_color = 'red'
        elif signal_state == 'yellow':
            signal_color = 'orange'
        elif signal_state == 'green':
            signal_color = 'green'
        else:
            signal_color = 'gray'
        return signal_color

def sim_plotting(df, label='dot', rate=0.3):
    # Time
    frames = df['Frame_ID'].unique()

    # Prepare plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_xlim(df['Local_X'].min() - 10, df['Local_X'].max() + 10)
    ax.set_ylim(df['Local_Y'].min() - 10, df['Local_Y'].max() + 10)
    ax.set_xlabel('Local_X')
    ax.set_ylabel('Local_Y')
    ax.set_title('Vehicle Positions')


    # Frame-by-frame plot
    for frame_id in frames:
        ax.clear()
        ax.set_xlim(df['Local_X'].min() - 10, df['Local_X'].max() + 10)
        ax.set_ylim(df['Local_Y'].min() - 10, df['Local_Y'].max() + 10)
        ax.set_xlabel('Local_X')
        ax.set_ylabel('Local_Y')
        frame_data = df[df['Frame_ID'] == frame_id]

        # Draw fixed vertical lines for lanes
        # Line segment range for Y top and bottom
        y_top_min = 80
        y_top_max = df['Local_Y'].max() + 10
        y_bottom_min = df['Local_Y'].min() - 10
        y_bottom_max = 40

        # Line segment range for X top and bottom
        x_top_min = 9
        x_top_max = df['Local_X'].max() + 10
        x_bottom_min = df['Local_X'].min() - 10
        x_bottom_max = -9

        # Draw vertical lines above Y > 70
        for x_line in [-9, -6, -3, 0, 3, 6, 9]:
            style = '-' if x_line == 0 or x_line == -9 or x_line == 9 else '--'
            ax.vlines(x=x_line, ymin=y_top_min, ymax=y_top_max,
                    color='gray', linestyle=style, linewidth=1)

        # Bottom lines (Y < 40)
        for x_line in [-9, -6, -3, 0, 3, 6, 9]:
            style = '-' if x_line == 0 or x_line == -9 or x_line == 9 else '--'
            ax.vlines(x=x_line, ymin=y_bottom_min, ymax=y_bottom_max,
                    color='gray', linestyle=style, linewidth=1)

        # Draw vertical lines above X > 70
        for y_line in [51, 54, 57, 60, 63, 66, 69]:
            style = '-' if y_line == 51 or y_line == 60 or y_line == 69 else '--'
            ax.hlines(y=y_line, xmin=x_top_min, xmax=x_top_max,
                    color='gray', linestyle=style, linewidth =1)

        # Bottom lines (X < 40)
        for y_line in [51, 54, 57, 60, 63, 66, 69]:
            style = '-' if y_line == 51 or y_line == 60 or y_line == 69 else '--'
            ax.hlines(y=y_line, xmin=x_bottom_min, xmax=x_bottom_max,
                    color='gray', linestyle=style, linewidth=1)
        
        # arrow for direction
        for x in [1.5, 4.5, 7.5]:
            ax.arrow(x, 10, 0, 10,  # from (x, 10) upward to (x, 20), vertical arrow to the top
                    head_width=0.5, head_length=3, fc='gray', ec='gray')
        for x in [-1.5, -4.5, -7.5]:
            ax.arrow(x, 110, 0, -10, # vertical arrow to the bottom
                    head_width=0.5, head_length=3, fc='gray', ec='gray')
        for y in [52.5, 55.5, 58.5]:
            ax.arrow(x=-15, y=y, dx=2, dy=0,  # horizontal arrow to the right
                    head_width=0.3, head_length=0.6,fc='gray', ec='gray')
        for y in [61.5, 64.5, 67.5]:
            ax.arrow(x=15, y=y, dx=-2, dy=0,  # horizontal arrow to the left
                    head_width=0.3, head_length=0.6,fc='gray', ec='gray')

        # Set signal line color
        signal_state_NS = frame_data['Signal_State_NS'].iloc[0]  # assume uniform per frame
        signal_state_EW = frame_data['Signal_State_EW'].iloc[0]
        signal_state_NSL = frame_data['Signal_State_NSL'].iloc[0]  # assume uniform per frame
        signal_state_EWL = frame_data['Signal_State_EWL'].iloc[0]
        signal_colar_NS = label_colar(signal_state_NS)
        signal_colar_EW = label_colar(signal_state_EW)
        signal_colar_NSL = label_colar(signal_state_NSL)
        signal_colar_EWL = label_colar(signal_state_EWL)

        # Define all signal lines as (type, position, range_min, range_max)
        signal_lines = [
            ('h', y_top_min, x_bottom_max, 0),        # top horizontal line (left side)
            ('h', y_bottom_max, 0, x_top_min),        # bottom horizontal line (right side)
            ('hl', y_top_min, x_bottom_max+6, 0),        # top horizontal line (left side)
            ('hl', y_bottom_max, 0, x_top_min-6),        # bottom horizontal line (right side)
            ('v', x_bottom_max, 51, 60),                 # right vertical line (northbound)
            ('v', x_top_min, 60, 69),               # left vertical line (southbound)
            ('vl', x_bottom_max, 51+6, 60),                 # right vertical line (northbound)
            ('vl', x_top_min, 60, 69-6)               # left vertical line (southbound)
        ]

        for orientation, pos, start, end in signal_lines:
            if orientation == 'h':
                ax.hlines(y=pos, xmin=start, xmax=end,
                        color=signal_colar_NS, linewidth=3, label='Signal')
            elif orientation == 'hl':
                ax.hlines(y=pos, xmin=start, xmax=end,
                        color=signal_colar_NSL, linewidth=3, label='Signal')
            elif orientation == 'v':
                ax.vlines(x=pos, ymin=start, ymax=end,
                        color=signal_colar_EW, linewidth=3, label='Signal')
            elif orientation == 'vl':
                ax.vlines(x=pos, ymin=start, ymax=end,
                        color=signal_colar_EWL, linewidth=3, label='Signal')
        
        # vehicle plot    
        if label == 'dot':
            ax.plot(frame_data['Local_X'], frame_data['Local_Y'], 'bo', markersize=5)
        else:
            # Plot text labels instead of dots
            for _, row in frame_data.iterrows():
                label = f"({int(row['Direction'])}, {int(row['Movement'])})"
                ax.text(row['Local_X'], row['Local_Y'], label, fontsize=8, ha='center', va='center')
       
        ax.set_title(f'Frame ID: {frame_id}')
        ax.set_aspect(aspect=rate)
        plt.pause(0.1)

    plt.ioff()
    plt.show()


if __name__ == "__main__":


    ## Simulation plotting
    # new_df = load_data(file_name='./simulation.csv')
    # print(len(new_df['Vehicle_ID'].unique()))
    # process_data_signal(file_name='./simulation.csv', out_filename='simulation2.csv')
    # sim_plotting(new_df)
    
    ## Data plotting
    # df = load_data(file_name='./sorted_data_final_selectwST.csv')
    # sim_plotting(df)

    ## Process data for simulation for vehicle first and then direction 
    ## and then XY for lane and then add signal timing 
    # process_data_4vehilce(file_name='./sorted_data.csv')
    # process_data_direction(file_name='./sorted_data_vehilce.csv')
    # process_data_XY(file_name='./sorted_data_direction.csv')
    process_data_signal(file_name='./sorted_data_final.csv')


    ## Simulation
    df = load_data(file_name='./sorted_data_final_selectwST.csv')
    dt = 0.1
    y_top_min = 80
    y_bottom_max = 40
    x_top_min = 9
    x_bottom_max = -9


    # Get unique frames
    frames = df['Frame_ID'].unique()

    for i in range(len(frames)-1):
        if i % 100 == 0:
            print("sim: ", i, "/", len(frames))
        # At each time frame
        frame_data = df[df['Frame_ID'] == frames[i]]
        # delete if out of boundary
        filtered_ids = frame_data[
            (((frame_data['Local_Y'] > 210) | (frame_data['Local_Y'] < -10)) &
            (frame_data['Direction'].isin([2, 4]))) |
            (((frame_data['Local_X'] > 35) | (frame_data['Local_X'] < -35)) &
            (frame_data['Direction'].isin([1, 3])))
        ]['Vehicle_ID'].unique()
        # Get the subset of df where Frame_ID > frames[i-1]
        rows_to_drop = df[
            (df['Frame_ID'] > frames[i-1]) &
            (df['Vehicle_ID'].isin(filtered_ids))
        ].index
        df = df.drop(rows_to_drop)
        df = df.reset_index(drop=True)
        # Reset index to preserve the original row positions 
        frame_data = frame_data.reset_index(drop=True)
        # find the lead vehicle of each lane
        first_indices = (
            frame_data.groupby('Lane_ID', sort=False)
                    .apply(lambda g: g.index[0])
                    .reset_index(name='Initial_Index')
        )
        # vehicle in this frame
        veh_id = frame_data['Vehicle_ID']
        # Update state with IDM
        for k in range(len(veh_id)):
            ## Direction: 
            # 4: South-Bound (SB) ↓  Local_Y decreasing; 
            # 2: North-Bound (NB) ↑  Local_Y increasing; 
            # 3: West-Bound (WB)  ←  Local_X decreasing; 
            # 1: East-Bound (EB)  →  Local_X increasing;
            if k not in first_indices['Initial_Index'].values:
                if frame_data['Direction'].iloc[k] == frame_data['Direction'].iloc[k-1]:
                    # IDM motion for signal timing
                    if frame_data['Direction'].iloc[k] in [2, 4]:
                        location_label = 'Local_Y'
                        ST_label = 'Signal_State_NS'
                        d = frame_data[location_label].iloc[k]
                        if frame_data['Direction'].iloc[k] == 4:
                            s_ST = d - y_top_min 
                        else:
                            s_ST = y_bottom_max - d  
                    else:
                        location_label = 'Local_X'
                        ST_label = 'Signal_State_EW'
                        d = frame_data[location_label].iloc[k]
                        if frame_data['Direction'].iloc[k] == 1:
                            s_ST = d - x_top_min 
                        else:
                            s_ST = x_bottom_max - d

                    v = frame_data['v_Vel'].iloc[k]
                    s = abs(frame_data[location_label].iloc[k-1] - frame_data[location_label].iloc[k])
                    delta_v = frame_data['v_Vel'].iloc[k] - frame_data['v_Vel'].iloc[k-1]
                    acc = IDM_with_signal(v, s, delta_v, signal_state=frame_data[ST_label].iloc[k],signal_distance=s_ST)
                    v = np.clip(v + acc * dt, 0, 16)
                    # Location update by direction
                    if frame_data['Direction'].iloc[k] in [1, 2]:
                        d = d + v * dt
                    elif frame_data['Direction'].iloc[k] in [3, 4]:
                        d = d - v * dt
                    # Find subjected vehicle ID in next frame and update 
                    dfindex = df[(df['Frame_ID'] == frames[i+1]) & (df['Vehicle_ID'] == veh_id[k])]
                    if len(dfindex) > 0:
                        index = dfindex.index[0]
                        df.at[index, 'v_Acc'] = acc
                        df.at[index, 'v_Vel'] = v
                        df.at[index, location_label] = d
    new_df = df[df['Frame_ID'].isin(frames)]
    # sorted_df = new_df.sort_values(by=['Vehicle_ID','Frame_ID']).reset_index(drop=True)
    
    new_df.to_csv('simulation.csv', index=False)
    

                    



        


        
        



  
    





