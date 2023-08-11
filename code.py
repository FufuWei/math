import csv
import math
import datetime
#user_habit_dict:每个用户的乘车记录:起点,终点,距离
user_habit_dict={}
#start_end_dict:每条记录的起点,终点对
start_end_dict={}
#end_start_dict:每条记录的起点,终点对
end_start_dict={}
#user_habit_dict_test:test中每个用户的记录
user_habit_dict_test={}
#bike_dict:bike中的记录
bike_dict={}

def rad(tude):
    return (math.pi/180.0)*tude

__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
__decodemap = { }
for i in range(len(__base32)):
    __decodemap[__base32[i]] = i
del i

def decode_exactly(geohash):
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    lat_err, lon_err = 90.0, 180.0
    is_even = True
    for c in geohash:
        cd = __decodemap[c]
        for mask in [16, 8, 4, 2, 1]:
            if is_even: # adds longitude info
                lon_err /= 2
                if cd & mask:
                    lon_interval = ((lon_interval[0]+lon_interval[1])/2, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], (lon_interval[0]+lon_interval[1])/2)
            else:      # adds latitude info
                lat_err /= 2
                if cd & mask:
                    lat_interval = ((lat_interval[0]+lat_interval[1])/2, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], (lat_interval[0]+lat_interval[1])/2)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon, lat_err, lon_err

def produceLocationInfo(latitude1, longitude1,latitude2, longitude2):
    radLat1 = rad(latitude1)
    radLat2 = rad(latitude2)
    a = radLat1-radLat2
    b = rad(longitude1)-rad(longitude2)
    R = 6378137
    d = R*2*math.asin(math.sqrt(math.pow(math.sin(a/2),2)+math.cos(radLat1)*math.cos(radLat2)*math.pow(math.sin(b/2),2)))
    detallat = abs(a)*R
    detalLon = math.sqrt(d**2-detallat**2)
    if b==0:
        direction = 1/2 if a*b>0 else -1/2
    else:
        direction = math.atan(detallat/detalLon*(1 if a*b>0 else -1))/math.pi
    return round(d)

def loc_2_dis(hotStartLocation,hotEndLocation):
    StartLocation = decode_exactly(hotStartLocation[:7])
    EndLocation = decode_exactly(hotEndLocation[:7])
    latitude1 = StartLocation[0]
    longitude1 = StartLocation[1]
    latitude2 = EndLocation[0]
    longitude2 = EndLocation[1]
    return produceLocationInfo(latitude1, longitude1, latitude2, longitude2)

def produceTimeInfo(TimeData):
    TimeData = TimeData.split(' ')
    baseData = datetime.datetime(2017, 5, 1, 0, 0, 1)
    mydata = TimeData[0].split('-')
    mytime = TimeData[1].split(':')
    mydata[0] = int(mydata[0])
    mydata[1] = int(mydata[1])
    mydata[2] = int(mydata[2])
    mytime[0] = int(mytime[0])
    mytime[1] = int(mytime[1])
    mytime[2] = int(mytime[2].split('.')[0])
    dt = datetime.datetime(mydata[0], mydata[1], mydata[2], mytime[0], mytime[1], mytime[2])
    minute = mytime[1]+mytime[0]*60
    # return int((dt-baseData).__str__().split(' ')[0]),miao,dt.weekday(),round(miao/900)
    isHoliday = 0
    if dt.weekday()in [5,6] or int((dt-baseData).__str__().split(' ')[0]) in [29,28]:
        isHoliday=1
    return isHoliday,minute,int((dt-baseData).__str__().split(' ')[0])

def add2result(result1,result2):
    for each in result2:
        if each in result1:
            result1[each] = min(result1[each] ,result2[each] )
        else:
            result1[each] = result2[each]
    return result1


    trainfile = r'D:\math\校赛\题目1\数学建模题目\train_morning.csv'
    tr = csv.DictReader(open(trainfile))
    for rec in tr:
        user = rec['userid']
        start = rec['geohashed_start_loc']
        end = rec['geohashed_end_loc']
        rec['isHoliday'], rec['minute'], rec['data'] = produceTimeInfo(rec['starttime'])
        if user in user_habit_dict:
            user_habit_dict[user].append(rec)
        else:
            user_habit_dict[user] = [rec]
        if start in start_end_dict:
            start_end_dict[start].append(rec)
        else:
            start_end_dict[start] = [rec]
        if end in end_start_dict:
            end_start_dict[end].append(rec)
        else:
            end_start_dict[end] = [rec]

    print('train done!')

    testfile = r'D:\math\校赛\题目1\数学建模题目\test.csv'
    te = csv.DictReader(open(testfile))
    for rec in te:
        user = rec['userid']
        bike = rec['bikeid']
        rec['isHoliday'], rec['minute'], rec['data'] = produceTimeInfo(rec['starttime'])
        if user in user_habit_dict_test:
            user_habit_dict_test[user].append(rec)
        else:
            user_habit_dict_test[user] = [rec]
        if bike in bike_dict:
            bike_dict[bike].append(rec)
        else:
            bike_dict[bike] = [rec]
    print("test done!")

    subfile = r'D:\math\校赛\题目1\数学建模题目\sub.csv'
    sub = open(subfile, 'w')
    iter1 = 0
    # AllhotLocSort = sorted(end_start_dict.items(), key=lambda d: len(d[1]), reverse=True)
    te1 = csv.DictReader(open(testfile))
    for rec in te1:
        iter1 += 1
        if iter1 % 10000 == 0:
            print(iter1 / 20000, '%', sep='')
        # testTime = timeSlipt(rec['minute'])
        rec['isHoliday'], rec['minute'], rec['data'] = produceTimeInfo(rec['starttime'])
        user1 = rec['userid']
        bikeid1 = rec['bikeid']
        order1 = rec['orderid']
        start1 = rec['geohashed_start_loc']
        hour1 = rec['minute'] / 60
        minute1 = rec['minute']
        isHoliday1 = rec['isHoliday']
        biketype1 = rec['biketype']
        data1 = rec['data']
        result = {}
        hotLoc = {}

    if user1 in user_habit_dict:
        for eachAct in user_habit_dict[user1]:
            start2 = eachAct['geohashed_start_loc']
            end2 = eachAct['geohashed_end_loc']
            hour2 = eachAct['minute'] / 60
            isHoliday2 = eachAct['isHoliday']
            biketype2 = eachAct['biketype']
            data2 = rec['data']
            dis = loc_2_dis(start1, start2)
            dis = min(dis, 1000)  # 1000
            qidian = qidianquan * (dis / 100) ** 2
            detalaTime = abs(hour2 - hour1) if abs(hour2 - hour1) < 12 else 24 - abs(hour2 - hour1)
            shijian = shijianquan * (detalaTime / 12 * 10) ** 2
            dayType = isHoliday2 - isHoliday1
            jiejia = jiejiaquan * (dayType * 10) ** 2  # ?
            biType = int(biketype2) - int(biketype1)
            bike = bikequan * (biType * 10) ** 2  # 0.5

            # return 欧式距离,南北方向距离,东西方向距离,曼哈顿距离,方向(-0.5:0.5)
            # test2train_dis = loc_2_dis(start1,end2)
            # train2train_dis = loc_2_dis(start2,end2)
            # dis_detal = min(abs(test2train_dis[3]-train2train_dis[3]),1000)  #1000
            # direction_detal = abs(test2train_dis[4]-train2train_dis[4])
            # direction_detal = direction_detal if direction_detal<0.5 else 1-direction_detal
            # jvli = 4 * (dis_detal/100)**2
            # fangxiang = 1 * (direction_detal/0.5*10)**2
            score = qidian + shijian + jiejia + bike  # jvli+fangxiang
            # print(qidian,shijian,jiejia,bike,jvli,fangxiang)
            if end2 in hotLoc:
                hotLoc[end2] += 1
            else:
                hotLoc[end2] = 1
            if end2 in result:
                if
            result[end2] > score:
            result[end2] = score
            else:
            result[end2] = score
            for each in hotLoc:
                result[each] = result[each] / (hotLoc[each] ** zhishu)  # 0
            for each in result:
                result[each] = math.sqrt(result[each])

    if user1 in user_habit_dict_test:
        resulttest = {}
        user_habit_dict_test[user1].sort(key=lambda x: x['data'] * 60 * 24 + x['minute'])
        xuhao = 0
        for i in range(len(user_habit_dict_test[user1]) - 1):
            if user_habit_dict_test[user1][i]['orderid'] == order1:
                xuhao = i
                resulttest[user_habit_dict_test[user1][i + 1]['geohashed_start_loc']] = 21
        for i in range(len(user_habit_dict_test[user1])):
            if i not in [xuhao, xuhao + 1]:
                resulttest[user_habit_dict_test[user1][i]['geohashed_start_loc']] = 21 + abs(i - xuhao)
            result = add2result(result, resulttest)

    if bikeid1 in bike_dict:
        resultleak = {}
        bike_dict[bikeid1].sort(key=lambda x: x['data'] * 60 * 24 + x['minute'])
        for i in range(len(bike_dict[bikeid1]) - 1):
            if bike_dict[bikeid1][i]['orderid'] == order1:
                zhong = bike_dict[bikeid1][i + 1]['data'] * 60 * 24 + bike_dict[bikeid1][i + 1]['minute']
                qi = bike_dict[bikeid1][i]['data'] * 60 * 24 + bike_dict[bikeid1][i]['minute']
                detal = zhong - qi
                if detal < 30:
                    resultleak[bike_dict[bikeid1][i + 1]['geohashed_start_loc']] = leak1
                elif detal < 2 * 60:
                    resultleak[bike_dict[bikeid1][i + 1]['geohashed_start_loc']] = leak2  # 4
                else:
                    resultleak[bike_dict[bikeid1][i + 1]['geohashed_start_loc']] = leak3  # 20
        result = add2result(result, resultleak)

    if start1 in start_end_dict:
    endDict = {}
    resultqizhong = {}
    for eachAct in start_end_dict[start1]:
        score = 0
        score += (24 - abs(hour1 - eachAct['minute'] / 60)) / 24
        score += (1 - abs(isHoliday1 - eachAct['isHoliday'])) * 0.4
        if eachAct['geohashed_end_loc'] in endDict:
            endDict[eachAct['geohashed_end_loc']] += score
        else:
            endDict[eachAct['geohashed_end_loc']] = score
    hotLoc = sorted(endDict.items(), key=lambda x: x[1], reverse=True)
    if len(hotLoc) >= 1:
        resultqizhong[hotLoc[0][0]] = 1000
    if len(hotLoc) >= 2:
        resultqizhong[hotLoc[1][0]] = 1001
    if len(hotLoc) >= 3:
        resultqizhong[hotLoc[2][0]] = 1002
    result = add2result(result, resultqizhong)

    for each in result:
        distance = loc_2_dis(each, start1)
        if distance > 2500:
            result[each] = 1999
    if start1 in result:
        result[start1] = min(2000, result[start1])
    else:
        result[start1] = 2000
    result['fuck2'] = 2001
    result['fuck3'] = 2002
    bestResult = sorted(result.items(), key=lambda d: d[1])
    string = rec['orderid']
    num = 0
    for item in bestResult:
        string += ',' + item[0]
        # string += ':' + str(item[1]) + '\t'
        num += 1
        if num == 3:
            break
    sub.write(string + '\n')

sub.close()
print('ok')

if __name__ == "__main__":
    training('train.csv', 'test.csv', 'submission.csv')






