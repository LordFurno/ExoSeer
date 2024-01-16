# import PySimpleGUI as sg

# """
#     Demo - Element List

#     All elements shown in 1 window as simply as possible.

#     Copyright 2022 PySimpleGUI
# """


# use_custom_titlebar = True if sg.running_trinket() else False

# def make_window(theme=None):

#     NAME_SIZE = 23


#     def name(name):
#         dots = NAME_SIZE-len(name)-2
#         return sg.Text(name + ' ' + '•'*dots, size=(NAME_SIZE,1), justification='r',pad=(0,0), font='Courier 10')

#     sg.theme(theme)

#     # NOTE that we're using our own LOCAL Menu element
#     if use_custom_titlebar:
#         Menu = sg.MenubarCustom
#     else:
#         Menu = sg.Menu

#     treedata = sg.TreeData()

#     treedata.Insert("", '_A_', 'Tree Item 1', [1234], )
#     treedata.Insert("", '_B_', 'B', [])
#     treedata.Insert("_A_", '_A1_', 'Sub Item 1', ['can', 'be', 'anything'], )

#     layout_l = [
#                 [name('Text'), sg.Text('Text')],
#                 [name('Input'), sg.Input(s=15)],
#                 [name('Multiline'), sg.Multiline(s=(15,2))],
#                 [name('Output'), sg.Output(s=(15,2))],
#                 [name('Combo'), sg.Combo(sg.theme_list(), default_value=sg.theme(), s=(15,22), enable_events=True, readonly=True, k='-COMBO-')],
#                 [name('OptionMenu'), sg.OptionMenu(['OptionMenu',],s=(15,2))],
#                 [name('Checkbox'), sg.Checkbox('Checkbox')],
#                 [name('Radio'), sg.Radio('Radio', 1)],
#                 [name('Spin'), sg.Spin(['Spin',], s=(15,2))],
#                 [name('Button'), sg.Button('Button')],
#                 [name('ButtonMenu'), sg.ButtonMenu('ButtonMenu', sg.MENU_RIGHT_CLICK_EDITME_EXIT)],
#                 [name('Slider'), sg.Slider((0,10), orientation='h', s=(10,15))],
#                 [name('Listbox'), sg.Listbox(['Listbox', 'Listbox 2'], no_scrollbar=True,  s=(15,2))],
#                 [name('Image'), sg.Image(sg.EMOJI_BASE64_HAPPY_THUMBS_UP)],
#                 [name('Graph'), sg.Graph((125, 50), (0,0), (125,50), k='-GRAPH-')]  ]

#     layout_r  = [[name('Canvas'), sg.Canvas(background_color=sg.theme_button_color()[1], size=(125,40))],
#                 [name('ProgressBar'), sg.ProgressBar(100, orientation='h', s=(10,20), k='-PBAR-')],
#                 [name('Table'), sg.Table([[1,2,3], [4,5,6]], ['Col 1','Col 2','Col 3'], num_rows=2)],
#                 [name('Tree'), sg.Tree(treedata, ['Heading',], num_rows=3)],
#                 [name('Horizontal Separator'), sg.HSep()],
#                 [name('Vertical Separator'), sg.VSep()],
#                 [name('Frame'), sg.Frame('Frame', [[sg.T(s=15)]])],
#                 [name('Column'), sg.Column([[sg.T(s=15)]])],
#                 [name('Tab, TabGroup'), sg.TabGroup([[sg.Tab('Tab1',[[sg.T(s=(15,2))]]), sg.Tab('Tab2', [[]])]])],
#                 [name('Pane'), sg.Pane([sg.Col([[sg.T('Pane 1')]]), sg.Col([[sg.T('Pane 2')]])])],
#                 [name('Push'), sg.Push(), sg.T('Pushed over')],
#                 [name('VPush'), sg.VPush()],
#                 [name('Sizer'), sg.Sizer(1,1)],
#                 [name('StatusBar'), sg.StatusBar('StatusBar')],
#                 [name('Sizegrip'), sg.Sizegrip()]  ]

#     # Note - LOCAL Menu element is used (see about for how that's defined)
#     layout = [[Menu([['File', ['Exit']], ['Edit', ['Edit Me', ]]],  k='-CUST MENUBAR-',p=0)],
#               [sg.T('PySimpleGUI Elements - Use Combo to Change Themes', font='_ 14', justification='c', expand_x=True)],
#               [sg.Checkbox('Use Custom Titlebar & Menubar', use_custom_titlebar, enable_events=True, k='-USE CUSTOM TITLEBAR-', p=0)],
#               [sg.Col(layout_l, p=0), sg.Col(layout_r, p=0)]]

#     window = sg.Window('The PySimpleGUI Element List', layout, finalize=True, right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_EXIT, keep_on_top=True, use_custom_titlebar=use_custom_titlebar)

#     window['-PBAR-'].update(30)                                                     # Show 30% complete on ProgressBar
#     window['-GRAPH-'].draw_image(data=sg.EMOJI_BASE64_HAPPY_JOY, location=(0,50))   # Draw something in the Graph Element

#     return window


# window = make_window()

# while True:
#     event, values = window.read()
#     # sg.Print(event, values)
#     if event == sg.WIN_CLOSED or event == 'Exit':
#         break

#     if values['-COMBO-'] != sg.theme():
#         sg.theme(values['-COMBO-'])
#         window.close()
#         window = make_window()
#     if event == '-USE CUSTOM TITLEBAR-':
#         use_custom_titlebar = values['-USE CUSTOM TITLEBAR-']
#         sg.set_options(use_custom_titlebar=use_custom_titlebar)
#         window.close()
#         window = make_window()
#     if event == 'Edit Me':
#         sg.execute_editor(__file__)
#     elif event == 'Version':
#         sg.popup_scrolled(__file__, sg.get_versions(), keep_on_top=True, non_blocking=True)
# window.close()
import numpy as np
import lightkurve as lk
from matplotlib import pyplot as plt
from scipy.stats import percentileofscore
def getBLSData(periodSearch, lightcurve):
    bls = lightcurve.to_periodogram(method='bls', period=periodSearch, frequency_factor=500)
    return bls

def extractPlanetData(blsData):
    planetPeriod = blsData.period_at_max_power
    planetT0 = blsData.transit_time_at_max_power
    planetDur = blsData.duration_at_max_power
    planetModel = blsData.get_transit_model(period=planetPeriod, transit_time=planetT0, duration=planetDur)
    return planetPeriod.value, planetT0.value, planetDur.value, planetModel

def createCadenceMask(blsData, period, t0, dur):
    cadenceMask = blsData.get_transit_mask(period=period, transit_time=t0, duration=dur)
    return cadenceMask

def applyCadenceMask(lightcurve, mask):
    maskedLc = lightcurve[~mask]
    return maskedLc

# Search for Kepler observations of Kepler-1000
search_result = lk.search_lightcurve('Kepler-1000', author='Kepler', exptime='long')

if list(search_result) != []:
    # Download all available Kepler light curves
    lc_collection = search_result.download_all()

lc=lc_collection.stitch().flatten(window_length=901).remove_outliers()

# Generate an array of periods
period = np.linspace(1, 100, 10000)

# Initialize a dictionary to store planet information
planets = {}

# Search for planets at different periods
counter = 1
#Figure out how to find multuple planets and cutoff threshold
# Set a maximum number of iterations to avoid an infinite loop
def findPlanets(period,lc):
    counter=0
    planets={}
    while counter <= 15:
        bls = getBLSData(period, lc)
        
        # Calculate the percentile of bls.power.max() in the entire distribution of bls.power
        medianRank=np.median(bls.power)
        
        data = extractPlanetData(bls)
        

        
        # Assuming you have a significance threshold for detection
        if bls.power.max() > medianRank*2.5:
            planets[counter] = [data[0], data[1], data[2]]
            mask = createCadenceMask(bls, planets[counter][0], planets[counter][1], planets[counter][2])
            lc = applyCadenceMask(lc, mask)
        else:
            break
        counter+=1
    return planets
        
print(findPlanets(period,lc))