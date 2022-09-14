#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:50:16 2022

@author: julienballbe
"""

# import numpy as np
# import pandas as pd


# from plotnine import ggplot, geom_line, aes, geom_abline, geom_point, geom_text, labels,geom_histogram,ggtitle

# import scipy
# from scipy.stats import linregress
# from scipy import optimize


# from plotnine.scales import scale_y_continuous,ylim,xlim,scale_color_manual
# from plotnine.labels import xlab,ylab
# from plotnine.coords import coord_cartesian



# from scipy.misc import derivative

# from scipy import signal

# import nest as nt
#%%
def start_NEST(module="New_Chizhov_module"):
    from pynestml.frontend.pynestml_frontend import generate_nest_target
    import nest
    import pandas as pd
    from plotnine import ggplot, geom_line,labs,xlim,ylim, aes, geom_abline,geom_point, geom_text, labels,geom_histogram,ggtitle
    import matplotlib.pyplot as plt
    
    import numpy as np
    
    generate_nest_target(input_path="/Users/julienballbe/My_Work/NEST/NEST_Models/New_Point_neuron_Chizhov.nestml",module_name=module,
    target_path="/tmp/nestml-target",logging_level='INFO')
    
    nest.Install(module)
    
#%%

def run_model_NEST(neuron_name,current_input,simulation_time,conductance_input=0.,C_m=100., G_l=10., v_l=-70., sigma_v=2., dt=0.001, do_plot=False):
    import nest
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution':dt})
    
    neuron=nest.Create(neuron_name)
    neuron.set({'s_input':conductance_input,
                'C_m':C_m,
                'G_l':G_l,
                'v_l':v_l,
                'sigma_v':sigma_v})
    
    voltmeter=nest.Create('voltmeter')
    voltmeter.set({"interval": dt})
    voltmeter.set({"record_from": ['V_m','v_th','delta_t']})
    current_generator=nest.Create('dc_generator')
    current_generator.set(amplitude=current_input)
    
    nest.Connect(current_generator,neuron)
    nest.Connect(voltmeter,neuron)
    
    sr = nest.Create('spike_recorder')
    nest.Connect(neuron, sr)
    
    nest.Simulate(simulation_time)
    membrane_potential_trace=pd.Series(voltmeter.get("events")["V_m"],name='Membrane_potential_mV',dtype='float64')
    time_trace=pd.Series(voltmeter.get("events")["times"],name='Time_ms',dtype='float64')
    spike_threshold_trace=pd.Series(voltmeter.get('events')['v_th'],name='Spike_threshold_potential',dtype='float64')
    spike_times=pd.Series(nest.GetStatus(sr, keys='events')[0]['times'],name='Spike_times_ms',dtype='float64')
    
    
    if do_plot:
        full_table=pd.concat([time_trace,membrane_potential_trace],axis=1)
        my_plot=ggplot(full_table,aes(x=full_table.iloc[:,0],y=full_table.iloc[:,1]))+geom_line()
        my_plot+=labs(title=str("Chizhov, s_input="+str(conductance_input)+"nS; C_m="+str(C_m)+"F; G_l="+str(G_l)+"nS; v_l="+str(v_l)+"mV; sigma_v="+str(sigma_v)+'mV'))
        
        # x = np.arange(0, 4 * np.pi, 0.1)
        # y = np.sin(x)
        # plt.plot(x, y)
        print(my_plot)
        
    return spike_times,membrane_potential_trace,time_trace
    

#%%

def model_F_I_curve_NEST(neuron_name,amplitude_start,amplitude_end,number_of_point,running_time,conductance_input=0.,C_m=100., G_l=10., v_l=-70., sigma_v=2., dt=0.001, do_plot=False):
    
    i_ramp=np.linspace(amplitude_start,amplitude_end,number_of_point)
    Stim_amp=pd.Series(i_ramp,name='Stim_amp_pA')
    Freq=np.array([])
    
    for current_i in i_ramp:
        
        
        current_spike_time=run_model_NEST(neuron_name=neuron_name,
                                          current_input=current_i,
                                          simulation_time=running_time,
                                          conductance_input=conductance_input,
                                          C_m=C_m,
                                          G_l=G_l,
                                          v_l=v_l,
                                          sigma_v=sigma_v,
                                          dt=dt,
                                          do_plot=False)[0]
        
        
        #current_spike_time=current_spike_time[current_spike_time>(3000-1000)]
        
        nb_spike=len(current_spike_time)

        frequency=nb_spike/((running_time)*1e-3)
        Freq=np.append(Freq,frequency)
    Freq=pd.Series(Freq,name='Frequency_Hz')
    F_I_table=pd.concat([Stim_amp,Freq],axis=1)
    F_I_table.columns=(['Stim_amp_pA','Frequency_Hz'])
    A = np.vstack([F_I_table[F_I_table.Frequency_Hz>0].iloc[:,0], np.ones_like(F_I_table[F_I_table.Frequency_Hz>0].iloc[:,0])]).T
    m, c = np.linalg.lstsq(A, F_I_table[F_I_table.Frequency_Hz>0].iloc[:,1], rcond=None)[0]
    
    if do_plot:
        my_plot=ggplot(F_I_table,aes(x=F_I_table.iloc[:,0],y=F_I_table.iloc[:,1]))+geom_point()#+xlim(0,160)+ylim(0,118)
        my_plot+=geom_abline(aes(intercept=c,slope=m))
        print(my_plot)
        
    return m,c
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    