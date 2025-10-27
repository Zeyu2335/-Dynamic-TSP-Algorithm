import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from simulation_baseline import *
import os
import math

def get_signal_state_NS1(frame_id):
    cycle_length = 1000
    phase = frame_id % cycle_length
    if 600 <= phase < 960:
        return 'green'
    elif 960 <= phase < 1000:
        return 'yellow'
    else:
        return 'red'
    

def get_signal_state_EW1(frame_id):
    cycle_length = 1000
    phase = frame_id % cycle_length

    if 0 <= phase < 560:
        return 'green'
    elif 560 <= phase < 590:
        return 'yellow'
    else:
        return 'red'

def frame_plotting(data, ax, bus_id, label='dot', rate=0.3):
    # Time
    frame_id = data[0, 0]
    
    # df column names: 0:'Frame_ID', 1:'Vehicle_ID', 2:'Lane_ID', 3:'Local_X', 4:'Local_Y', 
    # 5:'v_Vel', 6:'v_Acc', 7:'Movement', 8:'Direction', 9:'Signal_State_NS', 10:'Signal_State_EW'

    # Frame-by-frame plot
    ax.clear()
    ax.set_xlim(-50, 50)
    ax.set_ylim(-10, 210)
    ax.set_xlabel('Local_X')
    ax.set_ylabel('Local_Y')
    frame_data = data

    # Draw fixed vertical lines for lanes
    # Line segment range for Y top and bottom
    y_top_min = 80
    y_top_max = 210
    y_bottom_min = -50
    y_bottom_max = 40

    # Line segment range for X top and bottom
    x_top_min = 9
    x_top_max = 50
    x_bottom_min = -50
    x_bottom_max = -9

    # Draw vertical lines above Y > 70
    for x_line in [-9, -6, -3, 0, 3, 6, 9]:
        style = '-' if x_line == 0 or x_line == -9 or x_line == 9 or x_line == -3 else '--'
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
    signal_state_NS = frame_data[0, 9] # assume uniform per frame
    signal_state_EW = frame_data[0, 10]
    signal_colar_NS = label_colar(signal_state_NS)
    signal_colar_EW = label_colar(signal_state_EW)

    # Define all signal lines as (type, position, range_min, range_max)
    signal_lines = [
        ('h', y_top_min, x_bottom_max, 0),        # top horizontal line (left side)
        ('h', y_bottom_max, 0, x_top_min),        # bottom horizontal line (right side)
        ('v', x_bottom_max, 51, 60),                 # right vertical line (northbound)
        ('v', x_top_min, 60, 69)               # left vertical line (southbound)
    ]

    for orientation, pos, start, end in signal_lines:
        if orientation == 'h':
            ax.hlines(y=pos, xmin=start, xmax=end,
                    color=signal_colar_NS, linewidth=3, label='Signal')
        elif orientation == 'v':
            ax.vlines(x=pos, ymin=start, ymax=end,
                    color=signal_colar_EW, linewidth=3, label='Signal')
    
    # vehicle plot    
    if label == 'dot':
        for i in range(len(frame_data)):
            x, y = frame_data[i, 3], frame_data[i, 4]
            vehicle_id = int(frame_data[i, 1])

            # Example: Highlight a specific vehicle ID in red
            if vehicle_id == bus_id:  # <-- Replace with your target Vehicle_ID
                ax.plot(x, y, 'ro', markersize=6)  # red dot
            else:
                ax.plot(x, y, 'bo', markersize=5)  # blue dot
    else:
        # Plot text labels instead of dots
        for i in range(len(frame_data)):
            label = f"({int(frame_data[i, 8])}, {int(frame_data[i, 1])})"
            ax.text(frame_data[i, 3], frame_data[i, 4], label, fontsize=8, ha='center', va='center')
    
    ax.set_title(f'Time: {frame_id/10}', x=0.41)
    # ax.set_aspect(aspect=rate)
    plt.pause(0.01)

    
def df_arrival_direction(frames_end, arrival_info, seed=11):
    """
    Generate vehicle arrival times from a dictionary:
    {direction: (arrival_rate, num_vehicles)}

    Parameters:
    - frames: list or array of frame IDs
    - arrival_info: dict with direction as key and (rate, count) as value

    Returns:
    - DataFrame with ['Arrival_Time', 'Lane_ID', 'Direction', 'Vehicle_ID']
    """
    # np.random.seed(seed)
    labeled_arrivals = []
    for direction, (rate, count) in arrival_info.items():
        for lane_id in [2, 1, 11]:
            if lane_id == 11:
                arrivals = (generate_vehicle(arrival_rate=rate*0.1, num_vehicles=int(count*0.1)) * 10).astype(int)   
            else:
                arrivals = (generate_vehicle(arrival_rate=rate, num_vehicles=count) * 10).astype(int)   
            arrivals = np.unique(arrivals)
            labels = np.column_stack((
                arrivals,
                np.full(arrivals.shape, lane_id),
                np.full(arrivals.shape, direction)
            ))
            labeled_arrivals.append(labels)

    # Combine and sort
    combined = np.vstack(labeled_arrivals)
    combined = combined[combined[:, 0].argsort()]

    df_arrivals = pd.DataFrame(combined, columns=['Arrival_Time', 'Lane_ID', 'Direction'])

    # Filter arrivals beyond the last frame
    df_arrivals = df_arrivals[df_arrivals['Arrival_Time'] <= frames_end]
    df_arrivals['Vehicle_ID'] = np.array(range(len(df_arrivals)))

    return df_arrivals

def vehicle_direction_initial(direction, lane_id):
    """
    Get Local X and Y based on direction and lane_id
    - Direction 4: sort Local_Y ↓ 
    - Direction 2: sort Local_Y ↑ 
    - Direction 3: sort Local_X ← 
    - Direction 1: sort Local_X → 

    """
    xmax = 50
    xmin = -50
    ymax = 200
    ymin = 0
    if direction == 4:
        local_Y = ymax
        if lane_id == 2:
            local_X = -7.5
        elif lane_id == 1:
            local_X = -4.5
        elif lane_id == 11:
            local_X = -1.5
    elif direction == 2:
        local_Y = ymin
        if lane_id == 2:
            local_X = 7.5
        elif lane_id == 1:
            local_X = 4.5
        elif lane_id == 11:
            local_X = 1.5
    elif direction == 3:
        local_X = xmax
        if lane_id == 2:
            local_Y = 67.5
        elif lane_id == 1:
            local_Y = 64.5
        elif lane_id == 11:
            local_Y = 61.5
    elif direction == 1:
        local_X = xmin
        if lane_id == 2:
            local_Y = 58.5
        elif lane_id == 1:
            local_Y = 55.5
        elif lane_id == 11:
            local_Y = 52.5
    else:
        print('No such direction!')
    return local_X, local_Y

def calculate_times(mask, axis, first_df, last_df, v_desired):
    displacement = abs(last_df[mask][axis].values - first_df[mask][axis].values)
    t_free = displacement / v_desired
    t_actual = (last_df[mask]['Frame_ID'].values - first_df[mask]['Frame_ID'].values) / 10
    return t_actual, t_free

def simulate_queue(signal_plan, time_series, arrival_rate, discharge_rate=5, delta_t=0.1):
    queue = 0
    queue_over_time = []

    for t in time_series:
        signal = signal_plan[t]  # should be "green" or "red" or or "yellow"
        arrivals = arrival_rate * delta_t

        if signal == "red":
            queue += arrivals
        else:  # GREEN
            departures = min(queue, discharge_rate * delta_t)
            queue += arrivals - departures

        queue_over_time.append(round(queue, 2))

    return queue_over_time

def get_remaining_green(signal_list):
    remaining_list = []
    n = len(signal_list)
    for t in range(n):
        remaining = 0
        for i in range(t, n):
            if signal_list[i] in ['green', 'yellow'] :
                remaining += 1/10
            else:
                break
        remaining_list.append(remaining)
    return remaining_list

def calculate_green_extension(t_current, t_arrival, t_green_start, t_green_end, queue_length, dynamic,
                              headway=3, bus_clear_time=3,
                              min_extension=5.0, max_extension=7, extension_limit=20):
    """
    Calculate green extension time for TSP using conditional logic.
    
    Parameters:
    - t_arrival: Bus estimated arrival time (sec)
    - t_green_end: Current green phase end time (sec)
    - queue_length: Vehicles ahead of the bus
    - headway: Time gap between vehicles (default 2s)
    - bus_clear_time: Time needed for bus to cross intersection
    - min_extension: Minimum allowed green extension
    - max_extension: Maximum allowed green extension

    Returns:
    - green_extension: Final green extension time (sec)
    """

    # Total time needed: queue + bus
    t_needed = queue_length * headway + bus_clear_time
    green_type = 0
    if t_arrival > t_green_end + max_extension or t_arrival < t_green_start - max_extension:
        # Bus arrives after green ends or before green
        raw_extension = t_needed
        green_type = 1
    elif t_arrival <= t_green_start + 2 and t_arrival >= t_green_start - max_extension:
        # Bus close to green start
        raw_extension = t_arrival + t_needed - t_green_start
        green_type = 2
    elif t_arrival <= t_green_end + max_extension: 
        # Bus close to green end
        raw_extension = t_arrival + t_needed - t_green_end
        green_type = 3
    else:
        raw_extension = 0
        green_type  = 0
        print('No TSP needed!')

    if raw_extension <= 0:
        green_type = 0
        raw_extension = 0
        print('No TSP needed!')

    if dynamic == 'nodynm':
        if green_type in [1, 3]:
            if raw_extension > max_extension:
                raw_extension = 0
                green_type = 4
                print('TSP is not granted!')
    else:
        if green_type in [1, 3]:
            if raw_extension > extension_limit:
                raw_extension = 0
                green_type = 4
                print('TSP is not granted!')

            
    # Clip within min and max extension allowed
    
    if raw_extension != 0:
        if dynamic == 'nodynm':
            if raw_extension <= min_extension:
                green_extension = min_extension
            else:
                green_extension = max_extension
        else:
            if raw_extension <= min_extension:
                green_extension = min_extension
            else:
                green_extension = min(raw_extension, extension_limit)
        # green_extension = np.ceil(np.clip(raw_extension, min_extension, max_extension))
    else:
        green_extension = 0

    return green_extension, green_type

def apply_green_extension(bus_arrival_time, current_time, green_start, green_end, queue_clear_time, max_extension=7,min_extension = 5):
    """
    Extend the green if the bus is approaching and will arrive after current green ends.
    """
    green_type = 0
    if bus_arrival_time < green_start - 3:
        t_needed = bus_arrival_time + queue_clear_time - green_start
        if t_needed > 0:
            if t_needed <= min_extension:
                extension = min_extension
                green_type = 1
            elif t_needed <= max_extension:
                extension = max_extension
                green_type = 1
            else:
                green_type = 3
                extension = 0
                print('TSP is not granted')
        else:
            extension = 0
        # extension = np.ceil(max(0, min(t_needed, max_extension)))
        return extension, green_type
    elif bus_arrival_time + queue_clear_time <= green_end + max_extension + 3:
        # Bus is about to arrive but won't make it in time
        t_needed = bus_arrival_time + queue_clear_time - green_end
        if t_needed > 0:
            if t_needed <= min_extension:
                extension = min_extension
                green_type = 2
            elif t_needed <= max_extension:
                extension = max_extension
                green_type = 2
            else:
                green_type = 3
                extension = 0
                print('TSP is not granted')
        else:
            extension = 0
        # extension = np.ceil(max(0, min(t_needed, max_extension)))
        return extension, green_type
    else:
        return 0, green_type


if __name__ == "__main__":

    ## Direction: 
    # 4: South-Bound (SB) ↓  Local_Y decreasing; 
    # 2: North-Bound (NB) ↑  Local_Y increasing; 
    # 3: West-Bound (WB)  ←  Local_X decreasing; 
    # 1: East-Bound (EB)  →  Local_X increasing;

    # df_new = load_data(file_name='./simulation_baseline.csv')
    # sim_plotting(df_new, rate=0.6)

    plot_flag = False
    data_flow = 'user' # 'default' or 'user'
    ST_setting = 'TSP_rlc' #'TSP_rlc' or 'NoTSP' or 'TSP_extn'
    if ST_setting == 'TSP_rlc':
        payback = 'nopay' # 'pay', 'nopay'
        dynamic = 'dynm'  # 'dynm', 'nodynm'
        extension_limit = 10
    else:
        payback = ''
        dynamic = ''
    con  = 100 # 0 .. 100
    arrival_rate_NS = 0.15
    arrival_rate_EW = 0.01
    num_rep = 30

    if dynamic == 'dynm':
        max_extension = extension_limit
    else:
        max_extension = 7


    greenNS = 36
    yellowNS = 3
    redNS = 60
    red_clear = 1
    cycle = 100
    green_min = 5
    # np.random.seed(11)

    v_desired = 15
    dt = 0.1
    y_top_min = 80
    y_bottom_max = 40
    x_top_min = 9
    x_bottom_max = -9

    arrival_time = 10
    departure_time = 2

    
    t_free_N = 160/v_desired
    t_free_S = 80/v_desired
    t_free_EW = 58/v_desired

    passengers_per_bus = 1
    passengers_per_car = 1

    np.random.seed(12)

    df = load_data(file_name='./sorted_data_final_selectwST.csv')
    frames = list(range(len(df['Frame_ID'].unique())))
    # df column names: 0:'Frame_ID', 1:'Vehicle_ID', 2:'Lane_ID', 3:'Local_X', 4:'Local_Y', 
    # 5:'v_Vel', 6:'v_Acc', 7:'Movement', 8:'Direction', 9:'Signal_State_NS', 10:'Signal_State_EW', 11 :'MPR'
    column_name = df.columns
    
    
    # Vectorize the function to generate signal state of direction
    vectorized_signal_NS = np.vectorize(get_signal_state_NS1)
    vectorized_signal_EW = np.vectorize(get_signal_state_EW1)
    # Apply it
    Signal_State_NS0 = vectorized_signal_NS(frames)
    Signal_State_EW0 = vectorized_signal_EW(frames)


    num = 6000
    frames = frames[:num]
    Signal_State_NS = Signal_State_NS0[:num]
    Signal_State_EW = Signal_State_EW0[:num]
    Signal_State_EW1 = Signal_State_EW.copy()
    Signal_State_NS1 = Signal_State_NS.copy()
    remain_green_NS = get_remaining_green(Signal_State_NS)
    remain_green_NS = np.clip(np.floor(remain_green_NS), 0, 39)


    queue_length_list_NS = simulate_queue(Signal_State_NS, frames, arrival_rate = arrival_rate_NS, discharge_rate=1/departure_time)
    queue_length_list_EW = simulate_queue(Signal_State_EW, frames, arrival_rate = arrival_rate_EW, discharge_rate=1/departure_time)

    if data_flow != 'default':
        # Generate arrival time
        arrival_info = {
            4: (arrival_rate_NS, int(arrival_rate_NS*1000)),
            2: (arrival_rate_NS, int(arrival_rate_NS*1000)),
            3: (arrival_rate_EW, int(arrival_rate_EW*1000)),
            1: (arrival_rate_EW, int(arrival_rate_EW*1000))
        }
        df_arrivals = df_arrival_direction(frames[-1], arrival_info, seed=11)
    else:
        # flow of row data
        # 4: 0.18, 2: 0.15, 1: 0.013, 3: only 1 cars
        first_appearances = df.drop_duplicates(subset='Vehicle_ID', keep='first')

        first_appearances = first_appearances[['Frame_ID','Lane_ID','Direction','Vehicle_ID']]
        first_appearances.rename(columns={'Frame_ID': 'Arrival_Time'}, inplace=True)
        df_arrivals = first_appearances.reset_index(drop=True)
        arrival_rate_NS = ''
        arrival_rate_EW = ''
# time = df_arrivals[df_arrivals['Direction']==1]['Arrival_Time'].values
# t_d = np.diff(time)
# np.median(t_d /len(t_d))
        
    # df_arrivals = df_arrivals[(df_arrivals['Direction'] == 4)&(df_arrivals['Lane_ID'] != 0)]
    df_arrivals0 = df_arrivals.copy()
    del df, vectorized_signal_NS, vectorized_signal_EW, Signal_State_NS0, Signal_State_EW0

    arrival_rate0_NS = []
    time = df_arrivals[(df_arrivals['Direction'] == 4) & (df_arrivals['Lane_ID'] == 1)]['Arrival_Time'].values
    for i in range(9):
        time_i = time[(time > i*1000) & (time < (i+1)*1000)]
        arrival_rate0_NS.append(round(1/np.mean(np.diff(time_i)/10),2))
    
    frame_end = frames[-1]
    veh_id = df_arrivals0['Vehicle_ID']
    MPR_list = np.random.uniform(0, 1, size=len(veh_id))
    df_arrivals0['MPR'] = np.round(MPR_list, 1)
    veh_id_4 = df_arrivals[(df_arrivals['Direction'] == 4) & (df_arrivals['Lane_ID'] == 1)& (df_arrivals['Arrival_Time'] > frames[10]) & (df_arrivals['Arrival_Time'] < frames[-2000])]['Vehicle_ID'].values
    
    bus_id_list = np.random.choice(veh_id_4, num_rep, replace=False)
    bus_enter_list = [df_arrivals[df_arrivals['Vehicle_ID'] == bus_id]['Arrival_Time'].values[0] for bus_id in bus_id_list]   
    # bus_enter_list = bus_enter_list[::-1]
    # bus_id_list = bus_id_list[::-1]

    for con in [0, 20, 40, 60, 80, 100]:#[0, 20, 40, 60, 80, 100]
        vehicle_delay_list = []
        bus_delay_list = []
        green_extension_list = []
        queue_list = []
        arrival_list = []
        TSP_grant_list = []
        bus_pass_list = [] 
        arrival_actual_list = []
        total_delay = 0
        queue_length = 0
        arrival_time = 0
        bus_all_list = np.empty((0, 20))
        
        TSP_grant = False
        for rep in range(num_rep):
            Signal_State_EW = Signal_State_EW1.copy()
            Signal_State_NS = Signal_State_NS1.copy()
            df_arrivals = df_arrivals0.copy()
            bus_id = bus_id_list[rep]
            bus_enter = bus_enter_list[rep]
            
            vehicle_id_exist = np.array([])
            repeat = True
            bus_pass = False
            repeat2 = True

            delay = [[] for _ in range(4)]
            frame_data_update = None
            TSP_flag = False
            TSP_grant = False
            green_extension = 0
            bus_state = np.empty((0, 12))
            bus_all = np.empty((0, 20))

            # Prepare plot
            if plot_flag == True:
                plt.ion()  # Turn on interactive mode
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_xlim(-50, 50)
                ax.set_ylim(-10, 210)
                ax.set_xlabel('Local_X')
                ax.set_ylabel('Local_Y')
                ax.set_title('Vehicle Positions')
            
            for i in range(frame_end):
                if i % 2000 == 0:
                    print("sim: ", i, "/", frame_end+1)

                # Get new arrival vehicle and detect from the df_arrivals
                frame_data_new = None
                if frames[i] in df_arrivals['Arrival_Time'].values:
                    frame_data_new = []
                    new_arrivals = df_arrivals[df_arrivals['Arrival_Time'] == frames[i]]
                    # Add vehicle in veh_id_in
                    for j, row in new_arrivals.iterrows():
                        local_X, local_Y = vehicle_direction_initial(row['Direction'], row['Lane_ID'])
                        if row['Direction'] in [1, 3]:
                            new_row = [frames[i], row['Vehicle_ID'], row['Lane_ID'], local_X, local_Y,
                                v_desired, 0, 1, int(row['Direction']), Signal_State_NS[i], Signal_State_EW[i], row['MPR']]
                        else:
                            new_row = [frames[i], row['Vehicle_ID'], row['Lane_ID'], local_X, local_Y,
                                v_desired, 0, 1, int(row['Direction']), Signal_State_NS[i], Signal_State_EW[i], row['MPR']]
                        frame_data_new.append(new_row)
                    # drop the vehicle if it enter the link
                    frame_data_new = np.array(frame_data_new, dtype=object)
                    df_arrivals = df_arrivals.drop(df_arrivals[df_arrivals['Arrival_Time'] == frames[i]].index)

                
                if len(vehicle_id_exist) > 0:
                    if bus_id in vehicle_id_exist:
                        # ['v_Vel', 'v_Acc', 'Space_Headway', 'Time_Headway', 'Lane_ID_2',
                        # 'time_to_green', 'time_green_left', 'traffic_signal_Red', 'traffic_signal_upstream_Red', 
                        # 'Queue_status_no_Queued', 'dist_to_int', 'traffic_volume']
                        # df column names: 0:'Frame_ID', 1:'Vehicle_ID', 2:'Lane_ID', 3:'Local_X', 4:'Local_Y', 
    # 5:'v_Vel', 6:'v_Acc', 7:'Movement', 8:'Direction', 9:'Signal_State_NS', 10:'Signal_State_EW', 11 :'MPR'
                        row_idx = np.where(frame_data_update[:, 1] == bus_id)[0]
                        state = frame_data_update[row_idx,:]
                        d0 = state[0, 4]
                        bus_state = np.vstack([bus_state, state])
                        if i <= bus_enter+11:
                            h0 = d0 - frame_data_update[row_idx-1, 4][0]
                            state[0, 4] = d0 - y_top_min
                            # headway and time headway
                            state = np.append(state, [[h0]], axis=1)
                            state = np.append(state, [[h0/state[0, 5]]], axis=1)
                            # traffic volume
                            traffic = len(frame_data_update[
                                    (frame_data_update[:, 8] == 4) &
                                    (frame_data_update[:, 2] == 1) &
                                    (frame_data_update[:, 4] >= y_top_min) &
                                    (frame_data_update[:, 4] < d0)
                                    ])
                            state = np.append(state, [[traffic]], axis=1)
                            # time-to-green and green left
                            t2g =  i - (i // (cycle * 10) * cycle + redNS)
                            if t2g <= 0:
                                t2g = 0
                            gl = i // (cycle * 10) * cycle + redNS + greenNS + yellowNS - i
                            if t2g != 0:
                                gl = 0
                            state = np.append(state, [[t2g/10]], axis=1)
                            state = np.append(state, [[gl/10]], axis=1)
                            state = np.append(state, [[gl/10]], axis=1)
                            state = np.append(state, [[Signal_State_NS[i]]], axis=1)
                            state = np.append(state, [[Signal_State_NS[i]]], axis=1)
                            bus_all = np.vstack([bus_all, state])
                        if i == bus_enter+11:
                            # print(i)
                            green_start_next = None
                            t_green_start = i // (cycle * 10) * cycle + redNS
                            t_green_end = i // (cycle * 10) * cycle + redNS + greenNS + yellowNS
                            # queue_length = queue_length_list[int(i + arrival_time*10)]
                            arrival_time = np.round((d0 - y_top_min)/8.5, 1)
                            queue_length = np.round(queue_length_list_NS[int(i + arrival_time*10)]/2) 
                            queue_length0 = queue_length.copy()
                            queue_length_EW = int(queue_length_list_EW[int(i + arrival_time*10)]/2)
                            arrival_time = np.round((d0 - y_top_min - queue_length * 2)/8.5, 1)
                            if Signal_State_NS[i + int(arrival_time*10)] == 'green':
                                queue_length = 0
                            num_pre = len(frame_data_update[
                                    (frame_data_update[:, 8] == 4) &
                                    (frame_data_update[:, 2] == 1) &
                                    (frame_data_update[:, 4] >= y_top_min) &
                                    (frame_data_update[:, 4] <= d0)
                                    ]) - 1
                            if con == 100:
                                if Signal_State_NS[i+int(arrival_time*10)] == 'red':
                                    queue_length = len(frame_data_update[
                                        (frame_data_update[:, 8] == 4) &
                                        (frame_data_update[:, 2] == 1) &
                                        (frame_data_update[:, 4] >= y_top_min) &
                                        (frame_data_update[:, 4] <= d0)
                                        ]) - 1
                                else:
                                    queue_length = len(frame_data_update[
                                        (frame_data_update[:, 8] == 4) &
                                        (frame_data_update[:, 2] == 1) &
                                        (frame_data_update[:, 4] >= y_top_min) &
                                        (frame_data_update[:, 4] <= 200)&
                                        (frame_data_update[:, 5] <= d0)
                                        ]) -1
                                queue_length_EW = int(len(frame_data_update[
                                    (frame_data_update[:, 8] == 1) &
                                    (frame_data_update[:, 4] >= x_top_min) &
                                    (frame_data_update[:, 4] <= 50) &
                                    (frame_data_update[:, 5] <= 1)
                                    ]) / 3)
                                speed = frame_data_update[
                                    (frame_data_update[:, 8] == 4) &
                                    (frame_data_update[:, 2] == 1) &
                                    (frame_data_update[:, 4] >= y_top_min) &
                                    (frame_data_update[:, 4] <= 200) &
                                    (frame_data_update[:, 5] > 1), 5
                                    ]
                                min_val = np.min(speed)
                                max_val = np.max(speed)
                                average_speed = (min_val + max_val) / 2
                                if average_speed < 8 or average_speed > 9.5:
                                    average_speed = 8.5

                                if num_pre < 3:
                                    average_speed = 9.5
                                if num_pre > 5:
                                    average_speed = 8.5
                                if num_pre == 0:
                                    average_speed = 14

                                if Signal_State_NS[i+int(arrival_time*10)] == 'red':
                                    arrival_time = (d0- y_top_min - queue_length * 2) / 8.5
                                else:
                                    arrival_time = (d0 - y_top_min - queue_length * 2) / average_speed
                                if Signal_State_NS[i + int(arrival_time)*10] == 'green':
                                    if Signal_State_NS[i + int(arrival_time-2)*10] == 'red':
                                        queue_length = queue_length / 2
                                    else:
                                        queue_length = 0
                            elif 0 < con < 100:
                                
                                if Signal_State_NS[i+int(arrival_time*10)] == 'red':
                                    con_list = frame_data_update[
                                        (frame_data_update[:, 8] == 4) &
                                        (frame_data_update[:, 2] == 1) &
                                        (frame_data_update[:, 4] >= y_top_min) &
                                        (frame_data_update[:, 4] <= d0) &
                                        (frame_data_update[:, 11] <= con/100)
                                        ][:, 11]
                                    
                                    idx = np.where(con_list<=con/100)[0]
                                    idx_diff = np.diff(idx)
                                    if len(np.where(idx_diff>1)[0] > 1):
                                        queue_length = max(len(frame_data_update[
                                            (frame_data_update[:, 8] == 4) &
                                            (frame_data_update[:, 2] == 1) &
                                            (frame_data_update[:, 4] >= y_top_min) &
                                            (frame_data_update[:, 4] <= d0) &
                                            (frame_data_update[:, 11] <= con/100)
                                            ])-1, 0)
                                    else:
                                        queue_length = max(len(frame_data_update[
                                        (frame_data_update[:, 8] == 4) &
                                        (frame_data_update[:, 2] == 1) &
                                        (frame_data_update[:, 4] >= y_top_min) &
                                        (frame_data_update[:, 4] <= d0) &
                                        (frame_data_update[:, 11] <= con/100) 
                                        ])-1, 0)

                                else:
                                    queue_length = max(len(frame_data_update[
                                        (frame_data_update[:, 8] == 4) &
                                        (frame_data_update[:, 2] == 1) &
                                        (frame_data_update[:, 4] >= y_top_min) &
                                        (frame_data_update[:, 4] <= d0)&
                                        (frame_data_update[:, 5] <= 1) &
                                        (frame_data_update[:, 11] <= con/100)
                                        ]) -1, 0)
                                queue_length = min(queue_length, queue_length0)
                                queue_length_EW = int(len(frame_data_update[
                                    (frame_data_update[:, 8] == 1) &
                                    (frame_data_update[:, 4] >= x_top_min) &
                                    (frame_data_update[:, 4] <= 50) &
                                    (frame_data_update[:, 5] <= 1) &
                                    (frame_data_update[:, 11] <= con/100)
                                    ]) / 3 / (con/100)) 
                                speed = frame_data_update[
                                    (frame_data_update[:, 8] == 4) &
                                    (frame_data_update[:, 2] == 1) &
                                    (frame_data_update[:, 4] >= y_top_min) &
                                    (frame_data_update[:, 4] <= 200) &
                                    (frame_data_update[:, 5] > 1) &
                                    (frame_data_update[:, 11] <= con/100), 5
                                    ]
                                num_pre = int(queue_length/(con/100))
                                if len(speed) > 1:
                                    min_val = np.min(speed)
                                    max_val = np.max(speed)
                                    average_speed = (min_val + max_val) / 2
                                    if average_speed < 8:
                                        average_speed = 8.5
                                else:
                                    average_speed = 9.5
                                if num_pre < 3:
                                    average_speed = 9.5
                                if num_pre > 5:
                                    average_speed = 8.5
                                if num_pre == 0:
                                    average_speed = 14
                                    
                                if np.isnan(queue_length):
                                    queue_length = int(queue_length_list_NS[int(i + arrival_time*10)])
                                if np.isnan(queue_length_EW):
                                    queue_length_EW = int(queue_length_list_EW[int(i + arrival_time*10)])
                                if Signal_State_NS[i+int(arrival_time*10)] == 'red':
                                    arrival_time = (d0 - y_top_min - np.ceil(queue_length/(con/100)) * 2) / 8.5
                                else:
                                    arrival_time = (d0 - y_top_min - np.ceil(queue_length/(con/100)) * 2) / average_speed
                                if Signal_State_NS[i+int(arrival_time*10)] == 'green':
                                    queue_length = 0
                            arrival_time = np.round(arrival_time, 1)
                            queue_length = int(queue_length)
                            if queue_length > 0:
                                queue_status = 1
                            else:
                                queue_status = 1
                            print("Estimated Arrival:", arrival_time)
                            bus_all = np.append(bus_all, np.full((bus_all.shape[0], 1), queue_status), axis=1)

                            if ST_setting != 'NoTSP':
                                if TSP_flag == False:  
                                    t_arrival = (i/10 + arrival_time)
                                    if ST_setting == 'TSP_rlc':
                                        green_extension, green_type = calculate_green_extension(i/10, t_arrival, t_green_start, t_green_end, queue_length, dynamic, extension_limit=max_extension)
                                        green_extension = int(green_extension)
                                        # green_extension = 5
                                        t_arrival = int(t_arrival)
                                        min_phase = green_min + yellowNS + red_clear
                                        if green_extension != 0:
                                            TSP_grant = True
                                            extra = int(green_extension/2) -1
                                            if t_arrival >= t_green_end or t_arrival <= t_green_start+2:
                                                if t_arrival > t_green_end:
                                                    # Bus arrives after green ends
                                                    red_end = (i // (cycle * 10)+1) * cycle + redNS
                                                    red_start = (i // (cycle * 10)+1) * cycle 
                                                else:
                                                    # Bus arrives befor green 
                                                    red_end = i // (cycle * 10) * cycle + redNS
                                                    red_start = i // (cycle * 10) * cycle 
                                                    green_start_next = (i // (cycle * 10)) * cycle + redNS + greenNS 
                                                if red_end - t_arrival  <=  max_extension:
                                                    # too close to red end, start green earlier
                                                    green_extension = max(green_extension, max_extension)
                                                    Signal_State_NS[(red_end-green_extension)*10:red_end*10] = 'green'
                                                    Signal_State_EW[(red_end-green_extension-red_clear)*10:red_end*10] = 'red'
                                                    Signal_State_EW[(red_end-green_extension-yellowNS-red_clear)*10:(red_end-green_extension-red_clear)*10] = 'yellow'
                                                elif t_arrival - red_start < max_extension:
                                                    # too close to red start, close green later
                                                    green_extension = max(green_extension, max_extension)
                                                    Signal_State_NS[(t_green_end-yellowNS)*10:(t_green_end+red_clear)*10] = 'green'
                                                    Signal_State_NS[(t_green_end-yellowNS)*10:(t_green_end-yellowNS+green_extension)*10] = 'green'
                                                    Signal_State_NS[(t_green_end-yellowNS+green_extension)*10:(t_green_end+green_extension)*10] = 'yellow'
                                                    Signal_State_EW[t_green_end*10:(t_green_end+green_extension)*10] = 'red'
                                                else:
                                                    # in the middle
                                                    Signal_State_NS[(t_arrival+extra)*10-green_extension*10:(t_arrival+extra)*10] = 'green'
                                                    Signal_State_NS[(t_arrival+extra)*10:(t_arrival+extra)*10+yellowNS*10] = 'yellow'
                                                    Signal_State_EW[(t_arrival+extra)*10-green_extension*10-red_clear*10:(t_arrival+extra)*10+yellowNS*10+red_clear*10] = 'red'
                                                    Signal_State_EW[(t_arrival+extra)*10-green_extension*10-(yellowNS+red_clear)*10:(t_arrival+extra)*10-green_extension*10-red_clear*10] = 'yellow'
                                            else:
                                                t_arrival = int(t_arrival)
                                                # Bus arrives during green 
                                                Signal_State_NS[(t_green_end-yellowNS)*10:t_green_end*10] = 'green'
                                                Signal_State_NS[t_green_end*10:(t_green_end+green_extension)*10] = 'green'
                                                Signal_State_NS[(t_green_end+green_extension)*10:(t_green_end+green_extension+yellowNS)*10] = 'yellow'
                                                Signal_State_EW[(t_green_end-yellowNS)*10:(t_green_end+green_extension+yellowNS+red_clear)*10] = 'red'
                                            if green_start_next is None:
                                                green_start_next = (i // (cycle * 10)+1) * cycle + redNS + greenNS 
                                            if payback == 'pay':
                                                Signal_State_NS[(green_start_next-green_extension-yellowNS-red_clear)*10: (green_start_next-green_extension-red_clear)*10] = 'yellow'
                                                Signal_State_NS[(green_start_next-green_extension-red_clear)*10: (green_start_next+yellowNS)*10] = 'red'
                                                Signal_State_EW[(green_start_next-green_extension)*10: (green_start_next+yellowNS+red_clear)*10] = 'green'
                                            elif  payback == 'nopay' and queue_length_EW != 0:
                                                Signal_State_NS[(green_start_next-green_extension-yellowNS-red_clear)*10: (green_start_next-green_extension-red_clear)*10] = 'yellow'
                                                Signal_State_NS[(green_start_next-green_extension-red_clear)*10: (green_start_next+yellowNS)*10] = 'red'
                                                Signal_State_EW[(green_start_next-green_extension)*10: (green_start_next+yellowNS+red_clear)*10] = 'green'
                                            # if plot_flag == True:
                                            #     plotting_signal(frames, Signal_State_NS)
                                            #     plotting_signal(frames, Signal_State_EW)
                                            print(f'Green Extention: {green_extension},{green_type}')
                                    else:
                                        green_extension, add_phase = apply_green_extension(t_arrival, i/10, t_green_start, t_green_end, (queue_length+1)*departure_time, max_extension=7)
                                        if green_extension != 0:
                                            TSP_grant = True
                                            if add_phase == 2:
                                                start_idx = int((t_green_end - yellowNS) * 10)
                                                Signal_State_NS = np.insert(Signal_State_NS, start_idx, ['green'] * (green_extension * 10))
                                                Signal_State_EW = np.insert(Signal_State_EW, start_idx, ['red'] * (green_extension * 10))
                                            elif add_phase == 1:
                                                start_idx = int((t_green_start - green_extension) * 10)
                                                Signal_State_NS[start_idx:int(t_green_start*10)] = 'green'
                                                Signal_State_EW[start_idx:int(t_green_start*10)] = 'red'
                                                Signal_State_EW[start_idx-yellowNS*10:int((t_green_start-yellowNS)*10)] = 'yellow'
                                                
                                            # if payback == 'pay':
                                            #     start_idx = int((t_green_end + extension + 1) * 10)
                                            #     end_idx = int((t_green_end + 2 * extension + 1) * 10)
                                            #     indices = np.arange(start_idx, end_idx)
                                            #     Signal_State_NS = np.delete(Signal_State_NS, indices, axis=0)
                                            #     Signal_State_EW = np.delete(Signal_State_EW, indices, axis=0)
                                            # elif payback == 'nopay' and queue_length_EW != 0:
                                            #     start_idx = int((t_green_end + extension + 1) * 10)
                                            #     end_idx = int((t_green_end + 2 * extension + 1) * 10)
                                            #     indices = np.arange(start_idx, end_idx)
                                            #     Signal_State_NS = np.delete(Signal_State_NS, indices, axis=0)
                                            #     Signal_State_EW = np.delete(Signal_State_EW, indices, axis=0)
                                        else:
                                            print('No extension!')
                                        
                                    TSP_flag = True

                            
                                
                    # At each time frame
                    frame_data = frame_data_update.copy()
                    # find the lead vehicle of each lane
                    lane_ids = frame_data[:, 2] #Lane_ID
                    first_index_array = np.array([np.where(lane_ids == lane)[0][0] for lane in np.unique(lane_ids)])

                    # vehicle in this frame
                    frame_data_update = frame_data.copy()
                    frame_data_update[:, 0] = frames[i] #Frame_ID
                    frame_data_update[:, 9] = Signal_State_NS[i]
                    frame_data_update[:, 10] = Signal_State_EW[i]
                    # Update state with IDM
                    for k in range(len(frame_data)):
                        ## Direction: 
                        # 4: South-Bound (SB) ↓  Local_Y decreasing; 
                        # 2: North-Bound (NB) ↑  Local_Y increasing; 
                        # 3: West-Bound (WB)  ←  Local_X decreasing; 
                        # 1: East-Bound (EB)  →  Local_X increasing;

                        # IDM motion for signal timing
                        if frame_data[k, 8] in [2, 4]: #Direction
                            location_label = 4 #Local_Y
                            ST_label = 9 # Signal_State_NS
                            d = frame_data[k, location_label]
                            if frame_data[k, 8] == 4: #Direction
                                s_ST = d - y_top_min 
                            else:
                                s_ST = y_bottom_max - d  
                        else:
                            location_label = 3 #Local_X
                            ST_label = 10 #Signal_State_EW
                            d = frame_data[k, location_label]
                            if frame_data[k, 8] == 1: #Direction
                                s_ST = x_bottom_max - d
                            else:
                                s_ST = d - x_top_min
                        v = frame_data[k, 5]

                        # lead or following vehicle
                        if k in first_index_array:
                            acc = IDM_with_signal(v, s=0, delta_v=0, v0=v_desired, leading=False, signal_state=frame_data[k, ST_label], signal_distance=s_ST)

                        else:
                            if frame_data[k, 8] == frame_data[k-1, 8]:
                                s = abs(frame_data[k-1, location_label] - frame_data[k, location_label])
                                delta_v = frame_data[k, 5] - frame_data[k-1, 5]
                                acc = IDM_with_signal(v, s, delta_v, v0=v_desired, signal_state=frame_data[k, ST_label],signal_distance=s_ST)
                            else:
                                acc = IDM_with_signal(v, s=0, delta_v=0, v0=v_desired, leading=False, signal_state=frame_data[k, ST_label], signal_distance=s_ST)
                        
                        # Speed update
                        v = v + acc * dt
                        if v < 0:
                            acc, v = 0, 0
                        else:
                            v = min(round(v, 2), 16)
                        # Location update by direction
                        if frame_data[k, 8] in [1, 2]:
                            d = round(d + v * dt, 2)
                        elif frame_data[k, 8] in [3, 4]:
                            d = round(d - v * dt, 2)
                        # Update motion
                        frame_data_update[k, 6] = acc
                        frame_data_update[k, 5] = v
                        frame_data_update[k, location_label] = d
                        # bus pass or not
                        if bus_id == frame_data[k, 1] and repeat == True:
                            if v == 0 and frame_data[k, ST_label]=='red':
                                bus_pass = False
                                repeat = False
                                print('Bus do not pass!!')
                            else:
                                bus_pass = True
                        # actual arrival time
                        if bus_id == frame_data[k, 1] and repeat2 == True:
                            if v < 1 or 76 < d < 78:
                                arrival_time_actual = (i - bus_enter - 11)/10
                                print("Actual:",arrival_time_actual)
                                repeat2 = False
                    

                    ## Direction: 
                    # 4: South-Bound (SB) ↓  Local_Y decreasing; 
                    # 2: North-Bound (NB) ↑  Local_Y increasing; 
                    # 3: West-Bound (WB)  ←  Local_X decreasing; 
                    # 1: East-Bound (EB)  →  Local_X increasing;
                    Local_X = frame_data_update[:, 3]
                    Local_Y = frame_data_update[:, 4]
                    Direction = frame_data_update[:, 8]
                    # Create mask for vehicles out of range
                    mask_drop = (
                        ((Local_Y < y_bottom_max) & (Direction == 4)) | 
                        ((Local_Y > y_top_min) & (Direction == 2)) |
                        ((Local_X < x_bottom_max) & (Direction == 3)) |
                        ((Local_X > x_top_min) & (Direction == 1))
                    )
                    # Keep only rows that are NOT out of range
                    veh_drop = frame_data_update[mask_drop, 1]
                    for j in veh_drop:
                        t_actual = (i - df_arrivals0[df_arrivals0['Vehicle_ID']==j]['Arrival_Time'].values[0]) / 10
                        dir_i = frame_data_update[frame_data_update[:, 1] == j, 8][0]
                        if dir_i in [4]:
                            if j == bus_id:
                                delay_i = (t_actual - t_free_N) * passengers_per_bus
                                bus_delay = round(delay_i,2)
                            else:
                                delay_i = (t_actual - t_free_N) * passengers_per_car
                        elif dir_i in [2]:
                            delay_i = (t_actual - t_free_S) * passengers_per_car
                        else:
                            delay_i = (t_actual - t_free_EW) * passengers_per_car
                        delay[dir_i-1].append(round(delay_i,2))  


                    frame_data_update = frame_data_update[~mask_drop]
                
                # Update datafram and delet out of range vehicles and add new vehicles
                if len(vehicle_id_exist) > 0 or frame_data_new is not None:
                    if frame_data_new is not None:
                        if len(vehicle_id_exist) > 0:
                            for j in range(len(frame_data_new)):
                                # if same lane same direction
                                row_index = np.where(
                                    (frame_data_update[:, 2] == frame_data_new[j, 2]) & 
                                    (frame_data_update[:, 8] == frame_data_new[j, 8])
                                )[0]
                                if len(row_index) > 0:
                                    # same lane same direction
                                    # if distance is greater than 5 m
                                    if abs(frame_data_new[j, 4] - frame_data_update[row_index[-1], 4] + frame_data_new[j, 3] - frame_data_update[row_index[-1], 3]) > 5:
                                        frame_data_update = np.insert(frame_data_update, row_index[-1]+1, frame_data_new[j, :], axis=0)
                                    else:
                                        veh_drop = np.append(veh_drop, frame_data_new[j, 1])
                                else:
                                    # if same lane 
                                    if frame_data_new[j, 2] in frame_data_update[:, 2]:
                                        row_index = np.where(frame_data_update[:, 2] == frame_data_new[j, 2])[0]
                                        frame_data_update = np.insert(frame_data_update, row_index[-1]+1, frame_data_new[j, :], axis=0)
                                    else:
                                        frame_data_update = np.append(frame_data_update, [frame_data_new[j, :]], axis=0)
                            # add new vehicle and drop vehicles
                            vehicle_id_exist = np.concatenate([vehicle_id_exist, frame_data_new[:,1]])
                            vehicle_id_exist = vehicle_id_exist[~np.isin(vehicle_id_exist, veh_drop)]
                        else:
                            # add new vehicle if no vehicle in the link
                            frame_data_update = frame_data_new.copy()
                            vehicle_id_exist = np.concatenate([vehicle_id_exist, frame_data_new[:,1]])
                    else:
                        # drop vehicles when no new vehicles entering the link
                        vehicle_id_exist = vehicle_id_exist[~np.isin(vehicle_id_exist, veh_drop)]
                if frame_data_update is not None and plot_flag == True and bus_id in vehicle_id_exist:
                    frame_plotting(frame_data_update, ax, bus_id)
            
            if plot_flag == True:
                plt.ioff()
                plt.show()

            mean_delay = [np.mean(sublist) for sublist in delay]
            sum_delay = [np.sum(sublist) for sublist in delay]
            print(f"Bus delay: {round(bus_delay, 2)} seconds.")
            print(f"Total Person delay: {sum_delay} seconds.\n")

            vehicle_delay_list.append(mean_delay)
            bus_delay_list.append(bus_delay)
            green_extension_list.append(green_extension)
            bus_pass_list.append(bus_pass)
            queue_list.append(queue_length)
            arrival_list.append(arrival_time)
            arrival_actual_list.append(arrival_time_actual)
            TSP_grant_list.append(TSP_grant)
            total_delay = np.sum(sum_delay) + bus_delay * 19  + total_delay
        # if con == 0:
        #     bus_all_list = np.vstack([bus_all_list, bus_all])
        vehicle_delay_list = np.array(vehicle_delay_list)
        # print(f"Average Bus delay: {round(np.mean(bus_delay_list), 1)} ({round(np.std(bus_delay_list), 1)}) seconds.")
        # print(f"Average Total Person delay: {np.round(np.mean(vehicle_delay_list,axis=0), 1)} ({round(total_delay/num_rep,1)}) seconds.")
        # print(f"Average Green Extension: {round(np.mean(green_extension), 1)} ({round(np.std(green_extension), 1)}) seconds.")
        # print(f"Success pass: {sum(bus_pass_list)}")
        # print(f"TSP grant: {sum(TSP_grant_list)}")
        # print(f"Queue length: {np.mean(queue_list)}")
        # print(f"Arrival time: {np.mean(arrival_list)}")
        # print(f"Actual Arrival time: {np.mean(arrival_actual_list)}")
        
        # Combine into a DataFrame
        results_df = pd.DataFrame({
            'Vehicle_Delay1': vehicle_delay_list[:,0],
            'Vehicle_Delay2': vehicle_delay_list[:,1],
            'Vehicle_Delay3': vehicle_delay_list[:,2],
            'Vehicle_Delay4': vehicle_delay_list[:,3],
            'Bus_Delay': bus_delay_list,
            'Green_Extension': green_extension_list,
            'TSP grant':TSP_grant_list,
            'Bus pass':bus_pass_list,
            'queue': queue_list,
            'arrival time': arrival_list,
            'Actual arrival time': arrival_actual_list,
        })

        # Save to CSV
        # Ensure the folder exists
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)

        # Build full file path
        file_path = os.path.join(
            output_dir,
            f'simulation_results_{ST_setting}{payback}{max_extension}_{con}_{data_flow}_NS{arrival_rate_NS}_EW{arrival_rate_EW}.csv'
        )

        # Save DataFrame
        results_df.to_csv(file_path, index=False)
        # if con == 0:
        #     np.savetxt("bus.csv", bus_all_list, delimiter=",", fmt="%.2f")
