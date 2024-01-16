import PySimpleGUI as sg
import detection
import matplotlib.pyplot as plt
import numpy as np
def isValid(value,valueType):#Check if a value is valid or not
    if valueType=="keplerMission":
        if value.isnumeric()==False:
            return False
        if value.isnumeric()==True and int(value)>1000:
            return False
        if value.isnumeric()==True and int(value)<=0:
            return False
    elif valueType=="period":
        if value.isnumeric()==False:
            return False
        if value.isnumeric()==True and int(value)>1000:
            return False
        if value.isnumeric()==True and int(value)<=1:
            return False

    return True    
def clearErrors():
    window["-keplerMissionStatus-"].update("")
    window["-periodStatus-"].update("")
    window["-cadenceStatus-"].update("")

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
text="No error"
names=["long","short","fast"]
layout = [  [sg.Text("Kepler mission number:"), sg.InputText()],
            [sg.Text('Cadence time: '), sg.Combo(names, font=('Arial Bold', 12),  expand_x=True, enable_events=True,  readonly=True, key='-COMBO-')],
            [sg.Text('Period search: '),sg.InputText()],
            [sg.Button('Display light curve')],
            [sg.Button('Display BLS graph')],
            [sg.Button('Get planetary data')],
            [sg.Text("",font=('Arial Bold',15),key="-keplerMissionStatus-")],
            [sg.Text("",font=('Arial Bold',15),key="-periodStatus-")],
            [sg.Text("",font=('Arial Bold',10),key="-cadenceStatus-")]]



# Create the Window
lc=None
bls=None
planets={}#Planet: [period,t0,duration]
window = sg.Window('Window Title', layout,size=(550,300))
# Event Loop to process "events" and get the "values" of the inputs
while True:
    
    event, values = window.read()
    
    if values!=None:
        cadence=values["-COMBO-"]
        keplerMission=values[0]
        searchPeriod=values[1]
        if isValid(keplerMission,"keplerMission")==False:
            window["-keplerMissionStatus-"].update("Error: Please enter valid kepler mission (1-1000)")
        else:
            window["-keplerMissionStatus-"].update("")

        if isValid(searchPeriod,"period")==False:
            window["-periodStatus-"].update("Error: Please enter valid period to search (1-1000)")
        else:
            window["-periodStatus-"].update("")



        if event=="Display light curve" and isValid(keplerMission,"keplerMission") and isValid(searchPeriod,"period"):
            clearErrors()
            lc=detection.getLightcurveData("Kepler-"+str(keplerMission),cadence)
            if type(lc)==str:
                window["-cadenceStatus-"].update("Error: Cadence does not exsist, please change to another cadence.")
            else:
                lc.plot()
                plt.show(block=True)#Keeps it open

        if event=="Display BLS graph" and isValid(keplerMission,"keplerMission") and isValid(searchPeriod,"period"):
            period=np.linspace(1,int(searchPeriod),10000)
            clearErrors()
            if lc is not None and type(lc)!=str:
                    bls=detection.getBLSData(period,lc)
                    bls.plot()
                    plt.show(block=True)
                #ALready been calcualted
            else:
                #Fix this
                lc=detection.getLightcurveData("Kepler-"+str(keplerMission),cadence)
 
                if type(lc)==str:
                    window["-cadenceStatus-"].update("Error: Cadence does not exsist, please change to another cadence.")
                else:
                    bls=detection.getBLSData(period,lc)
                    bls.plot()
                    plt.show(block=True)
                #Need to get lc
        if event=="Get planetary data" and isValid(keplerMission,"keplerMission") and isValid(searchPeriod,"period"):
            clearErrors()
            period=np.linspace(1,int(searchPeriod),10000)
            if lc is not None and type(lc)!=str:
                print("Calculating planets")
                planets=detection.findPlanets(period,lc)
                print(f"Planets: {planets}")
            else:
                print("Getting light curve")
                lc=detection.getLightcurveData("Kepler-"+str(keplerMission),cadence)
 
                if type(lc)==str:
                    window["-cadenceStatus-"].update("Error: Cadence does not exsist, please change to another cadence.")
                else:
                    print("Calculating planets")
                    planets=detection.findPlanets(period,lc)
                    print(f"Planets: {planets}")





        #Need a text bos for errors (maybe status bar)
        print(list(values.items()))
        print(cadence)
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    # print('You entered ', values)

window.close()