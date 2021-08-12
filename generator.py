from os.path import exists as pathExists
from json import dump as jsonDump
from random import randint
from os import remove
import datetime

if pathExists('./adult2.data'):
    remove('./adult2.data')

rootDay = {}


def genDataOneCow(idIteration: int):
    # generate id
    id: int = idIteration + 1

    # generate age
    age: int = randint(0, 22)

    # generate weight
    if 0 <= age <= 7:
        weight = randint(200, 900)
    elif 7 < age <= 22:
        weight = randint(900, 1400)

    # generate type of feed
    typeOfFeed = randint(1, 4)

    # ammount eaten
    ammountEatenBreakfast = randint(20, 30)
    ammountEatenLunch = randint(20, 30)
    ammountEatenDinner = randint(20, 30)

    # generate time stamps
    currentDateTime = datetime.datetime.now()

    timestampBreakfast = {
        "year": currentDateTime.year,
        "month": currentDateTime.month,
        "day": currentDateTime.day,
        "hour": 6,
        "minute": 0,
        "second": 0
    }

    timestampLunch = {
        "year": currentDateTime.year,
        "month": currentDateTime.month,
        "day": currentDateTime.day,
        "hour": 12,
        "minute": 0,
        "second": 0
    }

    timestampDinner = {
        "year": currentDateTime.year,
        "month": currentDateTime.month,
        "day": currentDateTime.day,
        "hour": 18,
        "minute": 0,
        "second": 0
    }

    label = 1 if age > 7 else 0

    currentRootData = {
        "ID": id,
        "Age": age,
        "Weight": weight,
        "Type of Feed": typeOfFeed,
        "Ammount Eaten": {
            0: ammountEatenBreakfast,
            1: ammountEatenLunch,
            2: ammountEatenDinner
        },
        "Timestamp": {
            0: timestampBreakfast,
            1: timestampLunch,
            2: timestampDinner
        },
        "Label": label
    }

    return currentRootData


def genDataWeekCow():
    for i in range(24):
        rootDay.update({i: genDataOneCow(i)})


def numberToLetter(number):
    lTNdic = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }

    number = str(number)

    numberStr = ""

    for i in number:
        numberStr += lTNdic[int(i)] + "-"

    numberStr = numberStr[:-1]

    return numberStr


def writeToFile():
    file = open('adult2.data', 'a')

    for i in range(len(rootDay)):
        writeString = "{id}, {age}, {wgt}, {tOF}, {aE012}, {tsYMD0}, {tsHM0}, {tsS0}, {tsYMD1}, {tsHM1}, {tsS1}, {tsYMD2}, {tsHM2}, {tsS2}, {label_key}.\n".format(
            id = rootDay[i]["ID"],
            age = numberToLetter(rootDay[i]["Age"]),
            wgt = rootDay[i]["Weight"],
            tOF = numberToLetter(rootDay[i]["Type of Feed"]),
            aE012 = (((rootDay[i]["Ammount Eaten"][0] * 100) + rootDay[i]["Ammount Eaten"][1]) * 100) + rootDay[i]["Ammount Eaten"][2],
            tsYMD0 = numberToLetter((((rootDay[i]["Timestamp"][0]["year"] * 100) + rootDay[i]["Timestamp"][0]["month"]) * 100) + rootDay[i]["Timestamp"][0]["day"]),
            tsHM0 = numberToLetter((rootDay[i]["Timestamp"][0]["hour"] * 100) + rootDay[i]["Timestamp"][0]["minute"]),
            tsS0 = numberToLetter(rootDay[i]["Timestamp"][0]["second"]),
            tsYMD1 = numberToLetter((((rootDay[i]["Timestamp"][1]["year"] * 100) + rootDay[i]["Timestamp"][1]["month"]) * 100) + rootDay[i]["Timestamp"][2]["day"]),
            tsHM1 = numberToLetter((rootDay[i]["Timestamp"][1]["hour"] * 100) + rootDay[i]["Timestamp"][1]["minute"]),
            tsS1 = rootDay[i]["Timestamp"][1]["second"],
            tsYMD2 = (((rootDay[i]["Timestamp"][2]["year"] * 100) + rootDay[i]["Timestamp"][2]["month"]) * 100) + rootDay[i]["Timestamp"][1]["day"],
            tsHM2 = (rootDay[i]["Timestamp"][2]["hour"] * 100) + rootDay[i]["Timestamp"][2]["minute"],
            tsS2 = numberToLetter(rootDay[i]["Timestamp"][2]["second"]),
            label_key = rootDay[i]["Label"]
        )
        # print(writeString)
        file.write(writeString)
    
    file.close()


genDataWeekCow()

with open("cowData.json", "w") as jsonFile:
    jsonDump(rootDay, jsonFile)

writeToFile()
