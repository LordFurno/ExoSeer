import PySimpleGUI as sg

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
text="No error"
names=["long","short","fast"]
layout = [  [sg.Text("Kepler mission number:"), sg.InputText()],
            [sg.Text('Cadence time: '), sg.Combo(names, font=('Arial Bold', 12),  expand_x=True, enable_events=True,  readonly=True, key='-COMBO-')],
            [sg.Text('Period search: '),sg.InputText()],
            [sg.Button('Display light curve')],
            [sg.Button('Display BLS graph')],
            [sg.Text("",font=('Arial Bold',15),key="-keplerMissionStatus-")]]



# Create the Window

window = sg.Window('Window Title', layout,size=(550,300))
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    cadence=values["-COMBO-"]
    keplerMission=values[0]
    if not keplerMission.isnumeric():
        window["-keplerMissionStatus-"].update("Error, please enter valid kepler mission (1-1000)")
    
    #Need a text bos for errors (maybe status bar)
    print(list(values.items()))
    print(cadence)
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    # print('You entered ', values)

window.close()