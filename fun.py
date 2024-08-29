import os
import pickle
import pandas as pd
import numpy as np
import math
import itertools
import os
import pickle
import matplotlib.pyplot as plt


## plot
# get envelope by axtracting max values from window:
def extract_max_values(timeseries, compression_rate=1000):
    max_values = []
    num_windows = len(timeseries) // compression_rate

    for i in range(num_windows):
        window = timeseries[i /51200*compression_rate : (i + 1) /51200*compression_rate]
        max_value = max(window)
        max_values.append(max_value)

    return max_values

#smooth envelope:
def smooth(timeseries, window_size=100):
    box = np.ones(window_size)/window_size
    smooth_timeseries = np.convolve(timeseries, box, mode='same')
    return smooth_timeseries

#MAIN function to get smooth envelope:
def envelope_F(timeseries, window_size=100):
    max_val_list = extract_max_values(timeseries)
    smoothed_max_val_list = smooth(max_val_list, window_size)
    return smoothed_max_val_list
    

def plot_time_window(vib_data, start_t, end_t, comp_rate=1):
    cropped = vib_data[(vib_data.index > start_t) & (vib_data.index < end_t)]
    plt.plot(cropped.index*comp_rate, cropped)
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude ')
    plt.show()
    
    
def plot_timewindowing(timedata, vibdata, correction_val=0):
    """
    plot vibration record with widows of cut
    
    PARAMETERS:
    -----------
    timedata: list of tuples
    vibdata: pandas series
    correction_val: float, optional
    """
    
    samples = len(vibdata)  # total number of samples
    x = np.arange(samples) / fs  # array of timestamps
    record = plt.plot(x, vibdata)
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.title('Vibration Record with Entrance and Exits Marks')
    for i in range(len(timedata)):
        windowmark = plt.axvspan(timedata[i][0], timedata[i][1], alpha=0.3, color='red', label='actual cut')
        
    plt.rcParams['figure.figsize'] = [10, 4]
    plt.show()
    

    
    
def find_higher_vibration_segments(vibration_data, threshold=0.5):
    
    corr_val = 0.1 #padding for start and end of cut
    
    above_threshold = vibration_data > threshold
    
    start_indices = np.where(~above_threshold[:-1] & above_threshold[1:])[0] + 1
    end_indices = np.where(above_threshold[:-1] & ~above_threshold[1:])[0] + 1
    
    higher_vibration_segments = []
    
    for start_idx, end_idx in zip(start_indices, end_indices):
        start_time = start_idx/51200*1000 + corr_val  # Replace with your actual time values
        end_time = end_idx/51200*1000 - corr_val     # Replace with your actual time values
        
        if end_time-start_time >= 2: #2sec 
            higher_vibration_segments.append((start_time, end_time))
    
    return higher_vibration_segments



def find_consecutive_drop(time_series, threshold, compression_rate=1000):
    above_threshold = False
    count = 0
    index = None
    
    for i, value in enumerate(time_series):
        if not above_threshold and value >= threshold:
            above_threshold = True
        
        if above_threshold and value < threshold:
            count += 1
            if count == 1:
                index = i
        
        if above_threshold and value >= threshold:
            count = 0
    
        if count >= 10:
            break
    
    if count >= 10:
        return index*compression_rate/51200 #dropti
    else:
        return None #when failed

    

def plot_segment_withDefects(vib_data, start_t, end_t, defects_time, comp_rate=1):
    cropped = vib_data[(vib_data.index > start_t) & (vib_data.index < end_t)]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cropped.index*comp_rate, cropped)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Amplitude ')
    
    for i in range(len(defects_time)):
        windowmark = ax.axvspan(defects_time[i][0], defects_time[i][1], alpha=0.3, color='red', label='actual cut')

        
def plot_timewindowing(timedata, vibdata, correction_val=0):
    """
    plot vibration record with widows of cut
    
    PARAMETERS:
    -----------
    timedata: list of tuples (/ segments)
    """
    fs = 51200
    samples = len(vibdata)  # total number of samples
    x = np.arange(samples) / fs  # array of timestamps
    record = plt.plot(x, vibdata)
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude [dB FS]')
    plt.title('Vibration Record with Entrance and Exits Marks')
    for i in range(len(timedata)):
        windowmark = plt.axvspan(timedata[i][0], timedata[i][1], alpha=0.3, color='red', label=i)
        if i <= 3:
            annotation = plt.annotate('CC', xy=(timedata[i][0], 0.1))
        else:
            annotation = plt.annotate(i-4, xy=(timedata[i][0], 0.1))
    
    plt.rcParams['figure.figsize'] = [10, 4]
    plt.margins(x=0, y=0)
    plt.show()

    
def find_defect_windows(segments, segment_i):
    segment_i += 4 #skip CC (circle clean)
    segment = segments[segment_i]
    segment_length = segment[1]-segment[0]
    defects_time = segment_length/4+segment[0], segment_length/4*2+segment[0], segment_length/4*3+segment[0]
    
    #window time span of defect: 
    defect_span = segment_length/5/6 #span of defective cut  ### ADJUST
    #correction value: (factor of defect_span to normalise it)
    cor_val =  defect_span/2.2 #because tool is slightly early  ### ADJUST


    defects_start_end = [[defects_time[0]-defect_span/2-cor_val, defects_time[0]+defect_span/2-cor_val],
                         [defects_time[1]-defect_span/2-cor_val, defects_time[1]+defect_span/2-cor_val],
                         [defects_time[2]-defect_span/2-cor_val, defects_time[2]+defect_span/2-cor_val]]
    #print("segmentnumber: " + str(segment_i-4) + " defectstime:\n" + str(defects_start_end))
    return defects_start_end



def plot_spectro_segment_withDefects(vib_data, segment_i, comp_rate=1):
    
    defects_time = find_defect_windows(segments, segment_i)
    
    segment_i += 4 #skip CC (circle clean)
    segment = segments[segment_i]
    start_t, end_t = segment[0], segment[1]
    cropped = vib_data[(vib_data.index > start_t) & (vib_data.index < end_t)]
    fs = 51200

    fig, axs = plt.subplots(2, 1, figsize=(12*1.2, 8*1.2))
    #plt.rcParams['figure.dpi'] = 1500
    #plt.rcParams['agg.path.chunksize'] = 300
    
    #raw time domain
    axs[0].plot(cropped.index*comp_rate, cropped)
    axs[0].set_xlabel('Time [sec]')
    axs[0].set_ylabel('Amplitude ')
    fig.tight_layout()
    axs[0].margins(x=0, y=0)
    axs[0].set_title('raw Record')
    
    for i in range(len(defects_time)):
        windowmark = axs[0].axvspan(defects_time[i][0], defects_time[i][1], alpha=0.3, color='red', label='actual cut')
    
    # spectrogram
    axs[1].specgram(x=cropped, Fs=fs, xextent=(start_t,end_t))
    #axs[1].set_yscale('symlog')
    axs[1].set_ylabel('frequency [Hz]')
    axs[1].set_xlabel('time [sec]')
    axs[1].set_title('Spectrogram')
    
    plt.show
    

def open_pkl(timeseg_name):
    #safe plane timeseg as pickle
    path = os.path.join(os.getcwd(), "timeseg")
    path_file = os.path.join(path, timeseg_name)
    with open(path_file, "rb") as f:
        time_seg = pickle.load(f)
    return time_seg

#plane_context = open_pkl("context_0_0.pkl")
#print(len(plane_context))


def open_timeseg(timeseg_name):
    """and flat the nested list
    """
    #safe plane timeseg as pickle
    path = os.path.join(os.getcwd(), "timeseg")
    path_file = os.path.join(path, timeseg_name)
    with open(path_file, "rb") as f:
        time_seg = pickle.load(f)
    
    #flatten the list:
    flattened_list = [item for sublist in time_seg for item in sublist]
    flattened_list = [item for sublist in flattened_list for item in sublist]
    flattened_list
    
    return flattened_list





def normalize_values(values):
    min_val = min(values)
    max_val = max(values)
    normalized_values = [(val - min_val) / (max_val - min_val) for val in values]
    return normalized_values

def plot_parameters(ax, params_ranges, params_current):
    parameters = list(params_ranges.keys())

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-1, len(parameters))
    ax.axis('off')  # Turn off the axis

    for i, parameter_name in enumerate(parameters):
        possible_values = params_ranges[parameter_name]
        current_value = params_current[parameter_name]
        

        # Check if there is only one possible value
        if len(possible_values) == 1:
            # Display parameter name on the left without y-axis
            ax.text(-0.1, i, f"{parameter_name}: {possible_values[0]}", ha='right', va='center', fontsize=12)
        else:
            normalized_values = normalize_values(possible_values)
            
            # Check if 'Gegen1' is given instead of 'Gegen' and change into 'Gegen'
            if current_value == "Gegen1":
                current_value = 0
            if current_value == "Gegen":
                current_value = 0
            if current_value == "Gleich":
                current_value = 1

            current_normalized_value = (current_value - min(possible_values)) / (max(possible_values) - min(possible_values))

            # Plot horizontal line with ticks
            ax.hlines(y=i, xmin=0, xmax=1, color='black', linewidth=1)
            ax.yaxis.set_ticks([])  # Hide y-axis

            # Plot ticks for possible values
            ax.scatter(normalized_values, [i] * len(normalized_values), color='black', marker='|', s=100)
            
            # Plot dot for current value
            ax.scatter(current_normalized_value, i, color='red', marker='o', s=100)
            
            # Annotate current value above the dot
            if parameter_name == "direct":
                if current_value == 0:
                    current_value = "Gegen"
                if current_value == 1:
                    current_value = "Gleich"
                
                ax.annotate(current_value, 
                             (current_normalized_value, i), 
                             textcoords="offset points", 
                             xytext=(0,10), 
                             ha='center')
                
            else:
                ax.annotate(f'{current_value:.3f}', 
                             (current_normalized_value, i), 
                             textcoords="offset points", 
                             xytext=(0,10), 
                             ha='center')

            # Display parameter name on the left without y-axis
            ax.text(-0.1, i, parameter_name, ha='right', va='center', fontsize=12)

    ax.set_title('Context')

def plot_spectro_segment_withDefects_withContext(vib_data, segments, segment_i, params_ranges, params_current, comp_rate=1):
    defects_time = find_defect_windows(segments, segment_i)

    segment_i += 4  # skip CC (circle clean)
    segment = segments[segment_i]
    start_t, end_t = segment[0], segment[1]
    cropped = vib_data[(vib_data.index > start_t) & (vib_data.index < end_t)]
    fs = 51200

    fig, axs = plt.subplots(3, 1, figsize=(12 * 1.2, 8 * 1.2))

    # Plot raw time domain
    axs[0].plot(cropped.index * comp_rate, cropped)
    axs[0].set_xlabel('Time [sec]')
    axs[0].set_ylabel('Amplitude')
    axs[0].margins(x=0, y=0)
    axs[0].set_title('Raw Record')

    # Plot defects
    print(defects_time)
    for i in range(len(defects_time)):
        axs[0].axvspan(defects_time[i][0], defects_time[i][1], alpha=0.3, color='red', label='actual cut')

    # Plot spectrogram
    axs[1].specgram(x=cropped, Fs=fs, xextent=(start_t, end_t))
    axs[1].set_ylabel('Frequency [Hz]')
    axs[1].set_xlabel('Time [sec]')
    axs[1].set_title('Spectrogram')

    # Call the function to plot the third subplot
    plot_parameters(axs[2], params_ranges, params_current)

    plt.tight_layout()
    plt.show()
    
    
    
def get_segment_context_params(plane_context, segment_i):
    #####################################################
    #######################   ###########################
    #######################   ###########################
    #######################   ###########################
    #######################   ###########################
    #####################################################
    #######################   ###########################
    #####################################################
    """die tabellen werte (constants) müssen angepasst werden!!!
    """
    
    #cutting parameters:
    v_c_table, f_z_table = 157, 0.026 # m/min, mm/flute

    line_context= plane_context[segment_i]
    #0: a_e, 1: v_c, 2: f_z, 3: a_p, 4: d, 5: flutes, 6: gegen/gleich
    a_e = line_context[0] #in percent
    v_c_doe = line_context[1] #in perecent
    v_c = line_context[1]*v_c_table #in m/min
    f_z_doe = line_context[2] #in percent
    f_z = line_context[2]*f_z_table #in mm/tooth
    a_p = line_context[3] #in mm
    d   = line_context[4] # in mm
    flutes = line_context[5] # in #
    direct = line_context[6] # 0:gegen, 1:gleich

    n = int(v_c*v_c_table/math.pi/d*1000)
    f = int(f_z*f_z_table*n*flutes)

    # Example usage:
    params_ranges = {
        'a_e':   [0.2, 0.5, 0.8],  # in percent
        'v_c':   [0.8 * v_c_table, 1 * v_c_table, 1.2 * v_c_table],  # in m/min
        'f_z':   [0.8 * f_z_table, 1 * f_z_table, 1.2 * f_z_table],  # in mm/flute
        'a_p':   [1, 2],  # in mm
        'd':     [5],  # in mm
        'flutes': [2],  # number of flutes
        'direct': [0, 1]  # gegen/gleich
    }
    params_current = {
        'a_e': a_e,
        'v_c': v_c,
        'f_z': f_z,
        'a_p': a_p,
        'd': d,
        'flutes': flutes,
        'direct': direct
    }
    return params_ranges, params_current


def safeplots_dashboard(vib_data, segments, segment_i, params_ranges, params_current, comp_rate=1):
    defects_time = find_defect_windows(segments, segment_i)

    segment_i += 4  # skip CC (circle clean)
    segment = segments[segment_i]
    start_t, end_t = segment[0], segment[1]
    cropped = vib_data[(vib_data.index > start_t) & (vib_data.index < end_t)]
    fs = 51200

    fig, axs = plt.subplots(2, 1, figsize=(12 * 1.2, 8 * 1.2))

    # Plot raw time domain
    axs[0].plot(cropped.index * comp_rate, cropped)
    axs[0].set_xlabel('Time [sec]')
    axs[0].set_ylabel('Amplitude')
    axs[0].margins(x=0, y=0)
    axs[0].set_title('Raw Record')

    # Plot defects
    #print(defects_time)
    for i in range(len(defects_time)):
        axs[0].axvspan(defects_time[i][0], defects_time[i][1], alpha=0.3, color='red', label='actual cut')

    # Plot spectrogram
    axs[1].specgram(x=cropped, Fs=fs, xextent=(start_t, end_t))
    axs[1].set_ylabel('Frequency [Hz]')
    axs[1].set_xlabel('Time [sec]')
    axs[1].set_title('Spectrogram')

    # Call the function to plot the third subplot
    #plot_parameters(axs[2], params_ranges, params_current)
    
    #format current context
    if params_current["direct"] == "Gegen1" or params_current["direct"] == "Gegen":
        direct = 0
    elif params_current["direct"] == "Gleich":
        direct = 1
    else:
        print("danger!!!" + str(params_current["direct"]))


    #build filename
    filename = f'a_e:{round(params_current["a_e"],3)}v_c:{round(params_current["v_c"],3)}f_z:{round(params_current["f_z"],3)}a_p:{int(params_current["a_p"])}d:{int(params_current["d"])}flutes:{int(params_current["flutes"])}direct:{direct}'
    
    filename = filename.replace('.', '')  # Remove dots
    filename = filename.replace(':', '')  # Remove colons


    plt.tight_layout()
    plt.savefig(f'dashboardplots/{filename}.png')
    plt.close(fig)
    
    return filename


def select_vibdata(vib_data, segments, segment_i):
    segment_i += 4  # skip CC (circle clean)
    segment = segments[segment_i]
    start_t, end_t = segment[0], segment[1]
    cropped = vib_data[(vib_data.index > start_t) & (vib_data.index < end_t)]
    fs = 51200
    return cropped



def find_comparison_window(segments, segment_i):
    segment_i += 4 #skip CC (circle clean)
    segment = segments[segment_i]
    segment_length = segment[1]-segment[0]
    delta = segment_length/4/2 #delta for non defect
    
    non_defects_time = segment_length/4+segment[0]+delta, segment_length/4*2+segment[0]+delta, segment_length/4*3+segment[0]+delta
    
    defects_time = segment_length/4+segment[0], segment_length/4*2+segment[0], segment_length/4*3+segment[0]
    
    #window time span: 
    window_span = segment_length/5/6 #span of defective cut  ### ADJUST
    #correction value: (factor of window_span to normalise it)
    cor_val =  window_span/2.2 #because tool is slightly early  ### ADJUST

    
    # NONDEFECT
    non_defects_start_end = [[non_defects_time[0]-window_span/2-cor_val, non_defects_time[0]+window_span/2-cor_val],
                         [non_defects_time[1]-window_span/2-cor_val, non_defects_time[1]+window_span/2-cor_val],
                         [non_defects_time[2]-window_span/2-cor_val, non_defects_time[2]+window_span/2-cor_val]]

    #DEFECT
    defects_start_end = [[defects_time[0]-window_span/2-cor_val, defects_time[0]+window_span/2-cor_val],
                         [defects_time[1]-window_span/2-cor_val, defects_time[1]+window_span/2-cor_val],
                         [defects_time[2]-window_span/2-cor_val, defects_time[2]+window_span/2-cor_val]]
    
    return non_defects_start_end, defects_start_end



#######

def find_comparison_windows(segments, segment_i, seg_direct):
    segment_i += 4 #skip CC (circle clean)
    segment = segments[segment_i]
    segment_length = segment[1]-segment[0]
    delta = segment_length/4/2 #delta for non defect
    
    non_defects_time = segment_length/4+segment[0]+delta, segment_length/4*2+segment[0]+delta, segment_length/4*3+segment[0]+delta
    
    defects_time = segment_length/4+segment[0], segment_length/4*2+segment[0], segment_length/4*3+segment[0]
    
    
    #window time span: 
    #window_span = segment_length/5/6*2 #span of defective cut  OLD!!!        
    window_span = 0.766614583333336 #new, fixed timespan!              ### ADJUST
    
    #direction correction:
    if seg_direct == 'Gleich': # 1, Gleich???
        dir_cor_val = 0.1 # at Gleich the defect is later              ### ADJUST
    else: dir_cor_val = -0.15 #else Gegen                              ### ADJUST
    
    #correction value: (factor of window_span to normalise it)
    cor_val =  window_span*dir_cor_val #because tool is slightly early  

    
    # NONDEFECT
    non_defects_start_end = [[non_defects_time[0]-window_span/2+cor_val, non_defects_time[0]+window_span/2+cor_val],
                         [non_defects_time[1]-window_span/2+cor_val, non_defects_time[1]+window_span/2+cor_val],
                         [non_defects_time[2]-window_span/2+cor_val, non_defects_time[2]+window_span/2+cor_val]]

    #DEFECT
    defects_start_end = [[defects_time[0]-window_span/2+cor_val, defects_time[0]+window_span/2+cor_val],
                         [defects_time[1]-window_span/2+cor_val, defects_time[1]+window_span/2+cor_val],
                         [defects_time[2]-window_span/2+cor_val, defects_time[2]+window_span/2+cor_val]]
    
    return non_defects_start_end, defects_start_end


def safeplots_dashboard_v2(vib_data, segments, segment_i, params_ranges, params_current, comp_rate=1):
    seg_direct = params_current['direct']
    _, defects_time = find_comparison_windows(segments, segment_i, seg_direct)

    segment_i += 4  # skip CC (circle clean)
    segment = segments[segment_i]
    start_t, end_t = segment[0], segment[1]
    cropped = vib_data[(vib_data.index > start_t) & (vib_data.index < end_t)]
    fs = 51200

    fig, axs = plt.subplots(2, 1, figsize=(12 * 1.2, 8 * 1.2))

    # Plot raw time domain
    axs[0].plot(cropped.index * comp_rate, cropped)
    axs[0].set_xlabel('Time [sec]')
    axs[0].set_ylabel('Amplitude')
    axs[0].margins(x=0, y=0)
    axs[0].set_title('Raw Record')

    # Plot defects
    #print(defects_time)
    for i in range(len(defects_time)):
        axs[0].axvspan(defects_time[i][0], defects_time[i][1], alpha=0.3, color='red', label='actual cut')

    # Plot spectrogram
    axs[1].specgram(x=cropped, Fs=fs, xextent=(start_t, end_t))
    axs[1].set_ylabel('Frequency [Hz]')
    axs[1].set_xlabel('Time [sec]')
    axs[1].set_title('Spectrogram')

    # Call the function to plot the third subplot
    #plot_parameters(axs[2], params_ranges, params_current)
    
    #format current context
    if params_current["direct"] == "Gegen1" or params_current["direct"] == "Gegen":
        direct = 0
    elif params_current["direct"] == "Gleich":
        direct = 1
    else:
        print("danger!!!" + str(params_current["direct"]))


    #build filename
    filename = f'a_e:{round(params_current["a_e"],3)}v_c:{round(params_current["v_c"],3)}f_z:{round(params_current["f_z"],3)}a_p:{int(params_current["a_p"])}d:{int(params_current["d"])}flutes:{int(params_current["flutes"])}direct:{direct}'
    
    filename = filename.replace('.', '')  # Remove dots
    filename = filename.replace(':', '')  # Remove colons


    plt.tight_layout()
    plt.show()
    plt.savefig(f'dashboardplots/{filename}.png')
    plt.close(fig)
    
    return filename