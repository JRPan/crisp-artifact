import plotly.graph_objects as go
import numpy as np
import pandas as pd
import re
import plotly

import csv
import statistics
import os

def plot(hw, sim, hw_col, sim_col, title):
    # correl_co = np.corrcoef(hw[hw_col].to_list(), sim[sim_col].to_list())[0][1]
    # hw_copy = hw[hw_col].copy(deep=True)
    # for i in range(len(hw_copy)):
    #     if hw_copy[i] == 0:
    #         hw_copy[i] = sim[sim_col][i]
    # mae = np.mean(np.abs((hw[hw_col] - sim[sim_col]) / hw_copy)) * 100
    moe_annote = []
    symbals = ['circle', 'x']

    fig = go.Figure()
    count = 0
    for app in sim['label'].unique():
        correl_co = np.corrcoef(hw[hw['label'] == app][hw_col].to_list(), sim[sim['label'] == app][sim_col].to_list())[0][1]
        hw_copy = hw[hw['label'] == app][hw_col].copy(deep=True)
        for i in range(len(hw_copy)):
            if hw_copy.iloc[i] == 0:
                hw_copy.iloc[i] = sim[sim['label'] == app][sim_col].iloc[i]
        mae = np.mean(np.abs((hw[hw['label'] == app][hw_col] - sim[sim['label'] == app][sim_col]) / hw_copy)) * 100
        # moe_annote.append("{0}: Correl={1:.4f} MAE={2:.4f}".format(app, correl_co, mae))
        # annot = "{0}: Corr={1:.2f}%".format(app, correl_co * 100)
        annot = "{0}: MAPE={1:.2f}%".format(app, mae)
        moe_annote.append(annot)
        fig.add_trace(
            go.Scatter(
                x=hw[hw['label'] == app][hw_col],
                y=sim[sim['label'] == app][sim_col],
                mode="markers",
                marker=dict(size=10, symbol=symbals[count % 2]),
                name=annot,
                text=annot
            )
        )
        count += 1
    
    fig.update_layout(
        # title=title,
        xaxis_title="HW " + title,
        yaxis_title="Sim " + title,
    )
    fig.update_layout(showlegend=True)

    annote = ""
    for i in range(len(moe_annote)):
        annote += moe_annote[i]
        if i != len(moe_annote) - 1:
            # newline
            annote += "<br>"
    # top left
    # fig.add_annotation(
    #     x=0,
    #     y=1.2,
    #     xref="paper",
    #     yref="paper",
    #     text=annote,
    #     showarrow=False,
    #     font=dict(size=30, family='sans-serif'),
    #     # left align
    #     align="left",
    # )

    # draw 1
    fig.add_shape(
        dict(
            type="line",
            x0=1,
            y0=1,
            x1=hw[hw_col].values.max(),
            y1=hw[hw_col].values.max(),
            line=dict(color="Red", width=1),
        )
    )
    # margin
    fig.update_layout(
        xaxis=dict(
            titlefont=dict(size=25, color="black",family='sans-serif'),
            tickfont=dict(size=20, color="black",family='sans-serif'),
            type="log",
            autorange=True,
            gridcolor="gainsboro"
        ),
        yaxis=dict(
            title="Sim " + title,
            titlefont=dict(size=25, color="black", family='sans-serif'),
            tickfont=dict(size=20, color="black", family='sans-serif'),
            type="log",
            autorange=True,
            gridcolor="gainsboro"
        ),
        width=800, height=400,
        title_font_family="sans-serif",
        title_font_size=25,
        margin=dict(l=20, r=10, t=30, b=20),
        # legend to top
        legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0.01,
            font=dict(size=25, family='sans-serif'),
            # orientation="h",
            traceorder="reversed",
        ),
        plot_bgcolor="white",
    )
    # backgroun white with grid
    # fig.update_layout(
    #     plot_bgcolor="white",
    #     xaxis=dict(gridcolor="gainsboro"),
    #     yaxis=dict(gridcolor="gainsboro"),
    # )

    if "rate" in title:
        fig.update_xaxes(type="linear")
        fig.update_yaxes(type="linear")
    # fig.show()
    return fig


def get_csv_data_for_merge(filepath):
    all_named_kernels = {}
    stat_map = {}
    apps = []
    cached_apps = []
    configs = []
    cached_configs = []
    stats = []
    gpgpu_build_num = None
    gpgpu_build_nums = set()
    accel_build_num = None
    accel_build_nums = set()
    data = {}
    with open(filepath, "r") as data_file:
        reader = csv.reader(data_file)  # define reader object
        state = "start"
        for row in reader:  # loop through rows in csv file
            if len(row) != 0 and row[0].startswith("----"):
                state = "find-stat"
                continue
            if state == "find-stat":
                current_stat = row[0]
                stats.append(current_stat)
                state = "find-apps"
                continue
            if state == "find-apps":
                apps = row[1:]
                state = "process-cfgs"
                continue
            if state == "process-cfgs":
                if len(row) == 0:
                    if any_data:
                        cached_apps = apps
                        cached_configs = configs
                        for config in configs:
                            count = 0
                            for appargs_kname in apps:
                                first_delimiter = appargs_kname.find("--")
                                appargs = appargs_kname[:first_delimiter]
                                kname = appargs_kname[first_delimiter + 2 :]
                                if (
                                    current_stat == "GPGPU-Sim-build"
                                    and data[config][count] != "NA"
                                ):
                                    gpgpu_build_num = data[config][count][21:28]
                                    gpgpu_build_nums.add(gpgpu_build_num)
                                if (
                                    current_stat == "Accel-Sim-build"
                                    and data[config][count] != "NA"
                                ):
                                    accel_build_num = data[config][count][16:23]
                                    accel_build_nums.add(accel_build_num)
                                stat_map[
                                    kname + appargs + config + current_stat
                                ] = data[config][count]
                                count += 1
                    apps = []
                    configs = []
                    data = {}
                    state = "start"
                    any_data = False
                    continue
                else:
                    any_data = True
                    if accel_build_num != None and gpgpu_build_num != None:
                        full_config = (
                            row[0]
                            # + "-accel-"
                            # + str(accel_build_num)
                            # + "-gpgpu-"
                            # + str(gpgpu_build_num)
                        )
                    else:
                        full_config = row[0]
                    configs.append(full_config)
                    data[full_config] = row[1:]

    app_and_args = []
    for appargs_kname in cached_apps:
        first_delimiter = appargs_kname.find("--")
        appargs = appargs_kname[:first_delimiter]
        kname = appargs_kname[first_delimiter + 2 :]
        if appargs not in all_named_kernels:
            all_named_kernels[appargs] = []
            app_and_args.append(appargs)
        all_named_kernels[appargs].append(kname)

    # The assumption here is that every entry in each stats file is run with the same
    # git hash number, if not we are just going to warn and fail.
    if len(gpgpu_build_nums) > 1 or len(accel_build_nums) > 1:
        exit(
            "File {0} contains more than one gpgpu_build_num or accelsim_build_num - this assumes one stats file has one build num: gpgpu: {1}"
            + " accel: {2}".format(filepath, gpgpu_build_nums, accel_build_nums)
        )
    return (
        all_named_kernels,
        stat_map,
        app_and_args,
        cached_configs,
        stats,
        gpgpu_build_nums,
    )

def get_relaunch(file_name):
    _relaunch_map = {}
    file = open("./data/" + sets[0] + file_name, "r")
    # for each line in the file
    for line in file:
        segments = line.split("/")
        wl = segments[1]
        if wl == "gpgpu-sim-builds":
            continue
        print(segments)
        app_arg = segments[1] + "/" + segments[2]
        config = segments[3]
        print(segments[4])
        relaunch = segments[4].split(":")[1].replace("\n", "")
        _relaunch_map[app_arg+config] = relaunch
    return _relaunch_map

def normalize(df, config, group):
    normalized_df = pd.DataFrame()

    for c_app in df[group].unique():
        same_config = df[df[group] == c_app]
        normalized_to = same_config[same_config["g_config"] == config]

        same_config_num = same_config.select_dtypes(include=[np.number])
        # all the columns that are not numbers
        same_config_not_num = same_config.select_dtypes(exclude=[np.number])
        normalized_to_num = normalized_to.select_dtypes(include=[np.number])
        if(normalized_to_num.empty):
            normalized_to_num = pd.DataFrame(0, index=[0], columns=same_config_num.columns) 
        normalized_nums = same_config_num.div(normalized_to_num.iloc[0]).replace([np.inf, -np.inf], np.nan).fillna(0)

        normalized_same_app = pd.concat([same_config_not_num, normalized_nums], axis=1)
        normalized_df = pd.concat([normalized_df, normalized_same_app])


    assert(normalized_df.shape[0] == df.shape[0])
    assert(normalized_df.shape[1] == df.shape[1])
    return normalized_df

def get_base_kernels(filepath, base_kernels):
    stat_map = {}
    apps = []
    configs = []
    stats = []
    gpgpu_build_num = None
    gpgpu_build_nums = set()
    accel_build_num = None
    accel_build_nums = set()
    data = {}
    with open(filepath, "r") as data_file:
        reader = csv.reader(data_file)  # define reader object
        state = "start"
        for row in reader:  # loop through rows in csv file
            if len(row) != 0 and row[0].startswith("----"):
                state = "find-stat"
                continue
            if state == "find-stat":
                current_stat = row[0]
                stats.append(current_stat)
                state = "find-apps"
                continue
            if state == "find-apps":
                apps = row[1:]
                state = "process-cfgs"
                continue
            if state == "process-cfgs":
                if len(row) == 0:
                    if any_data:
                        cached_apps = apps
                        cached_configs = configs
                        for config in configs:
                            count = 0
                            for appargs_kname in apps:
                                first_delimiter = appargs_kname.find("--")
                                appargs = appargs_kname[:first_delimiter]
                                kname = appargs_kname[first_delimiter + 2 :]
                                if (
                                    current_stat == "GPGPU-Sim-build"
                                    and data[config][count] != "NA"
                                ):
                                    gpgpu_build_num = data[config][count][21:28]
                                    gpgpu_build_nums.add(gpgpu_build_num)
                                if (
                                    current_stat == "Accel-Sim-build"
                                    and data[config][count] != "NA"
                                ):
                                    accel_build_num = data[config][count][16:23]
                                    accel_build_nums.add(accel_build_num)
                                stat_map[
                                    kname + appargs + config + current_stat
                                ] = data[config][count]
                                if (config == "ORIN-SASS-VISUAL" or config == "ORIN-SASS") and data[config][count] != "NA" and current_stat == "gpu_tot_sim_cycle\s*=\s*(.*)":
                                    if appargs not in base_kernels:
                                        base_kernels[appargs] = []
                                    base_kernels[appargs].append(kname)
                                    # print(kname)
                                count += 1
                    apps = []
                    configs = []
                    data = {}
                    state = "start"
                    any_data = False
                    continue
                else:
                    any_data = True
                    if accel_build_num != None and gpgpu_build_num != None:
                        full_config = (
                            row[0]
                            # + "-accel-"
                            # + str(accel_build_num)
                            # + "-gpgpu-"
                            # + str(gpgpu_build_num)
                        )
                    else:
                        full_config = row[0]
                    configs.append(full_config)
                    data[full_config] = row[1:]
    return base_kernels

workloads = ["render_passes_2k", 
             ]
wl_to_name = {
    "pbrtexture_2k": "PT2",
    "pbrtexture_2k_lod0": "PT2_LOD0",
    "pbrtexture_4k": "PT4",
    "instancing_2k": "IT2",
    "instancing_2k_lod0": "IT2_LOD0",
    "instancing_4k": "IT4",
    "render_passes_2k": "SPL2",
    "render_passes_2k_lod0": "SPL2_LOD0",
    "render_passes_4k": "SPL4",
    "sponza_2k": "SPH2",
    "sponza_4k": "SPH4",
    "materials_2k": "MT2",
    "materials_4k": "MT4",
    "platformer_2k": "PL2",
    "platformer_4k": "PL4",
}
wl_to_name_no_res = {
    "pbrtexture": "PT",
    "instancing": "IT",
    "render_passes": "SPL",
    "sponza": "SPH",
    "materials": "MT",
    "platformer": "PL",
}

def config_to_name(name):
    name = name.replace("ORIN-SASS", "")
    name = name.replace("-VISUAL", "")
    name = name.replace("-concurrent", "con")
    name = name.replace("-dynamic_sm3", "-55")
    name = name.replace("-invalidate_l2", "-L2")
    name = name.replace("-best", "-sch")
    name = name.replace("-utility", "-u")
    name = name.replace("-slicer", "-sl")
    if not "con" in name:
        name = "seq" + name
    return name


    
patterns = ['', '/', '\\', 'x', '-', '|', '+', '.']

wl = 'render_passes_2k'
sets = ['']

sim_path = './'
drawcall = 0
global_id = 0
sim = pd.DataFrame(columns=[
    'app', 'label', 'config',
    'drawcall', 'l1_tex_access', 'l1_tex_hit', 'l1_global_access','cycle', 'vs', 'fs', 'l2_tex_read', 'l2_tex_hit', 'tot_cycle'])
hw = None

for dataset in sets:
    for wl in [
        "render_passes_2k",
        "render_passes_2k_lod0"
         ]:
        count = 0
        drawcall_map = {}
        print(sim_path + dataset + wl + ".csv")
        (
            all_named_kernels,
            stat_map,
            apps_and_args,
            configs,
            stats,
            gpgpu_build_nums,
        )= get_csv_data_for_merge(sim_path + dataset + wl + ".csv")
        thread_count = 0
        last_kernel = ''
        for app in apps_and_args:
            if 'NO_ARGS' not in app:
                continue
            for config in configs:
                if config != 'RTX3070-SASS-concurrent-fg-VISUAL':
                    continue
                for kernel in all_named_kernels[app]:
                    id = kernel.split('-')[-1]
                    if "VERTEX" in kernel:
                        drawcall_map[kernel] = int(id) // 2 
                        # global_id += 1
                        drawcall = drawcall_map[kernel]
                    else:
                        vertex_name = "MESA_SHADER_VERTEX_func0_main-" + str(int(id) - 1)
                        drawcall = drawcall_map[vertex_name]
                    if ('lod0' in wl):
                        drawcall += 24
                    # print(kernel, drawcall)
                    kernel_index = int(kernel.split('-')[-1]) - 1
                    if (wl == 'pbrtexture_2k'):
                        if(kernel_index >= 2):
                            break
                        
                    stat = '\\s+Total_core_cache_stats_breakdown\\[TEXTURE_ACC_R\\]\\[TOTAL_ACCESS\\]\\s*=\\s*(.*)'
                    l1_tex_read = (stat_map.get(kernel + app + config + stat, "NA").replace("NA", "0"))

                    stat = '\\s+Total_core_cache_stats_breakdown\\[TEXTURE_ACC_R\\]\\[MISS\\]\\s*=\\s*(.*)'
                    l1_tex_miss = (stat_map.get(kernel + app + config + stat, "NA").replace("NA", "0"))

                    stat = '\\s+Total_core_cache_stats_breakdown\\[TEXTURE_ACC_R\\]\\[HIT\\]\\s*=\\s*(.*)'
                    l1_tex_hit = (stat_map.get(kernel + app + config + stat, "NA").replace("NA", "0"))
                    # l1_tex_hit = int(l1_tex_read) - int(l1_tex_miss)
                    # l1_tex_read = int(l1_tex_read) + int(l1_tex_miss)

                    stat = 'gpu_sim_cycle\\s*=\\s*(.*)'
                    cycle = (stat_map.get(kernel + app + config + stat, "NA").replace("NA", "0"))

                    stat = 'gpu_tot_sim_cycle\\s*=\\s*(.*)'
                    tot_cycle = (stat_map.get(kernel + app + config + stat, "NA").replace("NA", "0"))
                    
                    stat = '\\s+Total_core_cache_stats_breakdown\\[GLOBAL_ACC_R\\]\\[TOTAL_ACCESS\\]\\s*=\\s*(.*)'
                    l1_global_access = (stat_map.get(kernel + app + config + stat, "NA").replace("NA", "0"))

                    stat = '\\s+L2_cache_stats_breakdown\\[TEXTURE_ACC_R\\]\\[TOTAL_ACCESS\\]\\s*=\\s*(.*)'
                    l2_tex_read = (stat_map.get(kernel + app + config + stat, "NA").replace("NA", "0"))

                    stat = '\\s+L2_cache_stats_breakdown\\[TEXTURE_ACC_R\\]\\[MISS\\]\\s*=\\s*(.*)'
                    l2_tex_miss = (stat_map.get(kernel + app + config + stat, "NA").replace("NA", "0"))

                    stat = '\\s+L2_cache_stats_breakdown\\[TEXTURE_ACC_R\\]\\[HIT\\]\\s*=\\s*(.*)'
                    l2_tex_hit = (stat_map.get(kernel + app + config + stat, "NA").replace("NA", 
                    "0"))
                    
                    l2_tex_hit = int(l2_tex_read) - int(l2_tex_miss)
                    
                    name = wl
                    if 'lod0' in wl:
                        label = "LoD OFF"
                    else:
                        label = "LoD ON"
                    # label = 'Vertex Count'
                    if (drawcall not in sim['drawcall'].values):
                        count += 1
                        # sim = sim.append({
                        #     'app': app,'drawcall': drawcall, 'l1_tex_access': 0, 'l1_tex_hit': 0, 
                        #     'l1_global_access': 0, 'cycle': 0, 'l2_tex_read': 0, 'l2_tex_hit': 0
                        #     }, ignore_index=True)
                        new_row = pd.DataFrame([pd.Series(0, index=sim.columns)])
                        sim = pd.concat([sim, new_row], ignore_index=True)
                        sim.iloc[-1, sim.columns.get_loc('app')] = name
                        sim.iloc[-1, sim.columns.get_loc('label')] = label
                        sim.iloc[-1, sim.columns.get_loc('drawcall')] = drawcall
                        sim.iloc[-1, sim.columns.get_loc('config')] = config

                    # update all sim data
                    sim.loc[sim['drawcall'] == drawcall, 'l1_tex_access'] += int(l1_tex_read)
                    sim.loc[sim['drawcall'] == drawcall, 'l1_tex_hit'] += int(l1_tex_hit)
                    sim.loc[sim['drawcall'] == drawcall, 'l1_global_access'] += int(l1_global_access)
                    sim.loc[sim['drawcall'] == drawcall, 'cycle'] += int(cycle)
                    sim.loc[sim['drawcall'] == drawcall, 'l2_tex_read'] += int(l2_tex_read)
                    sim.loc[sim['drawcall'] == drawcall, 'l2_tex_hit'] += int(l2_tex_hit)
                    sim.loc[sim['drawcall'] == drawcall, 'tot_cycle'] = max(int(tot_cycle), sim.loc[sim['drawcall'] == drawcall, 'tot_cycle'].values[0]) 
        hw_prof = pd.read_csv('./hw_run/renderdoc_profiling/{0}.csv'.format(wl.replace("_lod0", "")).replace("4k", "2k"))
        if('sponza_2k' in wl):
            hw_prof = hw_prof.iloc[391:]
        if('pbrtexture_2k' in wl):
            hw_prof = hw_prof.iloc[1:]
        if('demo_2k' in wl):
            hw_prof = hw_prof.iloc[4:]
        if('materials_2k' in wl):
            hw_prof = hw_prof.iloc[4:]
        if('instancing_2k' in wl):
            hw_prof = hw_prof.iloc[1:]
        if('platformer_2k' in wl):
            hw_prof = hw_prof.iloc[2237:]
        hw_prof = hw_prof[:count]
        if hw is None:
            hw = hw_prof
        else:
            hw = pd.concat([hw, hw_prof], ignore_index=True)    

# sort sim based on drawcall
sim = sim.sort_values(by=['drawcall'])

sim = sim.fillna(0)
sim['l1_tex_hitrate'] = sim['l1_tex_hit'] / sim['l1_tex_access']
sim['l2_tex_hitrate'] = sim['l2_tex_hit'] / sim['l2_tex_read']
hw['label'] = sim['label'].to_list()


fig = plot(hw, sim,"l1tex__texin_requests.sum", "l1_tex_access", "L1 TEX Request")
fig.write_image("./{0}.pdf".format("l1_tex_lod"), format="pdf")