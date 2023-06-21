from tkinter import *
import numpy as np
import random
import copy
from collections import deque
import time
import os
class Agent:
    def __init__(self):
        self._preLoc = (1, 1)
        self._curLoc = (1, 1)  # 현재위치
        self._curDir = 0  # 현재방향 0: East North West Soutth
        self._isAlive = True  # 살았는지
        self._arrowCnt = 2  # 화살갯수

explore = np.array([[' X ', ' X ', ' X ', ' X ', ' X ', ' X '],
                    [' X ', ' ? ', ' ? ', ' ? ', ' ? ', ' X '],
                    [' X ', ' ? ', ' ? ', ' ? ', ' ? ', ' X '],
                    [' X ', ' ? ', ' ? ', ' ? ', ' ? ', ' X '],
                    [' X ', ' ? ', ' ? ', ' ? ', ' ? ', ' X '],
                    [' X ', ' X ', ' X ', ' X ', ' X ', ' X ']])  # 탐험중정보
exscreen = np.array([[' X ', ' X ', ' X ', ' X ', ' X ', ' X '],
                    [' X ', ' ? ', ' ? ', ' ? ', ' ? ', ' X '],
                    [' X ', ' ? ', ' ? ', ' ? ', ' ? ', ' X '],
                    [' X ', ' ? ', ' ? ', ' ? ', ' ? ', ' X '],
                    [' X ', ' ? ', ' ? ', ' ? ', ' ? ', ' X '],
                    [' X ', ' X ', ' X ', ' X ', ' X ', ' X ']])  # 탐험중정보 screen출력용도
# world = np.array([[' X ', ' X ', ' X ', ' X ', ' X ', ' X '],
#                    [' X ', ' O ', ' O ', ' O ', ' O ', ' X '],
#                    [' X ', ' O ', ' O ', ' O ', ' O ', ' X '],
#                   [' X ', ' O ', ' O ', ' O ', ' O ', ' X '],
#                   [' X ', ' O ', ' O ', ' O ', ' O ', ' X '],
#                   [' X ', ' X ', ' X ', ' X ', ' X ', ' X ']])  # 실제월드
# world2 = np.array ([[' X ', ' X ', ' X ', ' X ', ' X ', ' X '],
#                     [' X ', ' S ', ' O ', ' O ', ' G ', ' X '],
#                     [' X ', ' W ', ' S ', ' O ', ' O ', ' X '],
#                     [' X ', ' S ', ' O ', ' B ', ' O ', ' X '],
#                     [' X ', ' O ', ' B ', ' P ', ' B ', ' X '],
#                     [' X ', ' X ', ' X ', ' X ', ' X ', ' X ']])
# world = np.flip(world2, axis=-2)
outt = deque()
percept = np.array([' None ', ' None ', ' None  ', 'None', ' None'])
direction =(-1, 0), (0, -1), (1, 0), (0, 1)  # 상하좌우
Gold_flag = False
safe_zone = np.zeros((6, 6))  # bfs가 갈수있는곳인지 확인여부(안전한곳인지)

def init_World():  # 월드 생성
    # Gold, Wumpus, Pitch 무조건 하나씩 생성
    GWP = random.sample(range(2, 16), 3)  # 2에서 15중에서 3개 숫자 임의추출
    G = divmod(GWP[0], 4)
    Gx, Gy = G[0] + 1, G[1] + 1
    W = divmod(GWP[1], 4)
    Wx, Wy = W[0] + 1, W[1] + 1
    P = divmod(GWP[2], 4)
    Px, Py = P[0] + 1, P[1] + 1
    world[Gx][Gy] = ' G '
    world[Wx][Wy] = ' W '
    for i in direction:
        x = Wx + i[0]
        y = Wy + i[1]
        xy = world[x][y]
        if xy != ' W ' and xy != ' P ' and xy != ' G ' and xy != ' X ':
            if xy == ' B ': 
                world[x][y] = 'SB '
            else : 
                world[x][y] = ' S '
    world[Px][Py] = ' P '
    for i in direction:
        x = Px + i[0]
        y = Py + i[1]
        xy = world[x][y]
        if xy != ' W ' and xy != ' P ' and xy != ' G ' and xy != ' X ':
            if xy == ' S ':
                world[x][y] = 'SB '
            else:
                world[x][y] = ' B '
    # 1/10 확률로 Wumpus, Pitch 생성
    for i in range(16):
        if i == 1 or i == GWP[0] or i == GWP[1] or i == GWP[2]: continue  # 금이라면
        x = int(i / 4) + 1  # 0에서 15까지 숫자를 좌표료 변경
        y = int(i % 4) + 1
        global numW, numP
        numW, numP = 0, 0
        rW = random.choices(range(2), weights=[9, 1])[0]  # 10퍼 확률로 W생성
        rP = random.choices(range(2), weights=[9, 1])[0]  # 10퍼 확률로 P생성
        xs=world[x][y]
        if xs != ' G ' and xs!= ' P ' and xs!= ' X ' and xs!= ' W ':
            if rW:
                world[x][y] = ' W '
                for j in direction:
                    xx = x + j[0]
                    yy = y + j[1]
                    xy = world[xx][yy]
                    if xy != ' W ' and xy != ' P ' and xy != ' G ' and xy != ' X ':
                        if xy == ' B ':
                            world[xx][yy] = 'SB '
                        else:
                            world[xx][yy] = ' S '
                numW += 1
            elif rP:
                world[x][y] = ' P '
                for j in direction:
                    xx = x + j[0]
                    yy = y + j[1]
                    xy = world[xx][yy]
                    if xy != ' W ' and xy != ' P ' and xy != ' G ' and xy != ' X ':
                        if xy == ' S ':
                            world[xx][yy] = 'SB '
                        else:
                            world[xx][yy] = ' B '
                numP += 1

check = np.zeros((6, 6))  # agent가 가본곳인지 여부
safe_q = deque()  # dfs 출력하기위해 큐로이용


def percept_percept(loc):  # world에서 percept를 받는 함수
    global Gold_flag, safe_zone,ag
    percept = np.array([' None ', ' None ', ' None  ', 'None', ' None'])
    now = world[loc[0]][loc[1]]
    if now == ' G ':
        ag._curLoc = loc
        percept[2] = 'Glitter'
        Gold_flag = True
        print_agent(loc, percept)
        # sys.exit("금을찾음")
        #print("금을찾음!")
        outt.append(["Find gold!"])
        outt.append(["Grab"])
        #print("Grab")
        trace = bfs(ag._curLoc,(1,1),None)
        #print("집으로 가는 길")
        outt.append(["way home"])
        trace.append((1,1))
        if trace:
            for i in range(len(trace)):
                pass
                #print(trace[i])
        outt.append(trace)
        #print("Climb")
        percept = np.array([' None ', ' None ', ' None  ', 'None', ' None'])
        print_agent((1,1), percept)
        outt.append(["Climb"])
        print_outt(outt)
        os.system('pause')
        quit()
    elif now == ' W ' or now == ' P ':
        explore[loc[0]][loc[1]] = now
        exscreen[loc[0]][loc[1]] = now
        ag._isAlive = False  # Agent는 죽고 상태 초기화 후 다시 살아난다
        #print(now, loc, 'Dead')
        outt.append([now, loc, 'Dead'])

    elif now == ' X ':
        percept[3] = 'Bump'
    else:
        safe_zone[loc[0]][loc[1]] = 1
        if now == 'SB ':
            percept[0] = 'Stench'
            percept[1] = 'Breeze'
        elif now == ' S ':
            percept[0] = 'Stench'
        elif now == ' B ':
            percept[1] = 'Breeze'
    return percept


def print_agent(loc, percept):
    global ag,outt
    #print('Agent is in', (abs(loc[0]),abs(loc[1])))
    outt.append(["Agent is go to"])
    if percept[3] != 'Bump':
        #print(percept)
        global explore2
        explore2 = copy.deepcopy(exscreen)
        explore2[loc[0]][loc[1]] = ' A '
        #print(np.flip(explore2, axis=-2))
        outt.append([loc,percept, np.flip(explore2, axis=-2)])
    else:
        #print(percept)
        outt.append([loc,percept])
    #print()


def reasoning(loc, percept):
    global ag
    explore[loc[0]][loc[1]] = world[loc[0]][loc[1]]  # 월드를 explore에 그대로 update
    exscreen[loc[0]][loc[1]] = world[loc[0]][loc[1]]
    if percept[0] == 'Stench' and percept[1] == 'Breeze':  # SB일 때는 ?만 바뀔수있는 경우의 수입니다
        S_update(loc[0], loc[1])  # S나 B 주변이 변경되었을때 S,B를 다시 탐색하는 함수
        B_update(loc[0], loc[1])
        for i in direction:
            x = loc[0] + i[0]
            y = loc[1] + i[1]
            if explore[x][y] == ' ? ':  # 여기서 ?일때 WP?로 업데이트해주기만 하면됨
                explore[x][y] = 'WP?'
                exscreen[x][y] = 'WP?'
        return
    elif percept[0] == 'Stench':
        S_update(loc[0], loc[1])
        for i in direction:  # 상하좌우로 돈다
            x = loc[0] + i[0]
            y = loc[1] + i[1]
            if explore[x][y] == ' ? ' or explore[x][y] == 'WP?':
                explore[x][y] = ' W?'
                exscreen[x][y] = ' W?'
            elif explore[x][y] == ' P?':  # Stench옆에 P?가 있을때
                explore[x][y] = ' O '
                exscreen[x][y]=' O '
                for j in direction:
                    x1 = x + j[0]
                    y1 = y + j[1]
                    if explore[x1][y1] == ' S ':
                        S_update(x1, y1)  # S나 B 주변이 변경되었을때 S,B를 다시 탐색하는 함수
                    elif explore[x1][y1] == ' B ':
                        B_update(x1, y1)
                safe_q.append((x, y))
        return
    elif percept[1] == 'Breeze':  # Stench일때와 비슷하게 돌아갑니다
        B_update(loc[0], loc[1])
        for i in direction:
            x = loc[0] + i[0]
            y = loc[1] + i[1]
            if explore[x][y] == ' ? ' or explore[x][y] == 'WP?':
                explore[x][y] = ' P?'
                exscreen[x][y] = ' P?'
            elif explore[x][y] == ' W?':
                for j in direction:
                    x1 = x + j[0]
                    y1 = y + j[1]
                    if explore[x1][y1] == ' S ':
                        S_update(x1, y1)
                    elif explore[x1][y1] == ' B ':
                        B_update(x1, y1)
                safe_q.append((x, y))
        return
    else:
        explore[loc[0]][loc[1]] = ' O '  # O일 때 상하좌우를 집어넣는다(안전)
        for i in direction:
            x = loc[0] + i[0]
            y = loc[1] + i[1]
            if explore[x][y] == ' S ':
                S_update(x, y)  # S나 B 주변이 변경되었을때 S,B를 다시 탐색하는 함수
            elif explore[x][y] == ' B ':
                B_update(x, y)

            if explore[x][y] == ' X ':  # 벽일 때
                safe_q.append((-x, -y))
                percept2 = copy.deepcopy(percept)
            else:
                safe_q.append((x, y))
                #explore[x][y] = ' O '


def S_update(x1, y1):
    global ag
    not_w = [' P ', ' P?', 'SB ', ' S ', ' B ', ' O ', ' X ']
    may_w = [' W ', ' W?', 'WP?', ' ? ']
    f, f2 = False, False  # f이고 not f2일 때 단하나 있는것(주변 S or B 주변에 하나만 may_w있다면 확정시키기)
    for k in direction:  # 상하좌우
        x2 = x1 + k[0]
        y2 = y1 + k[1]
        for l in may_w:
            if explore[x2][y2] == l and f == False:
                f = (x2, y2)
            elif explore[x2][y2] == l:
                f2 = True
    if f and (not f2):
        explore[f[0]][f[1]] = ' W '  # 확정
        exscreen[f[0]][f[1]] = ' W '  # 확정
        global Wx, Wy
        Wx = f[0]  # Wumpus x,y 좌표
        Wy = f[1]


def B_update(x1, y1):
    global ag
    not_p = [' W ', ' W?', 'SB ', ' S ', ' B ', ' O ', ' X ']
    may_p = [' P ', ' P?', 'WP?', ' ? ']
    f, f2, f3 = False, False, False  # 주변 S or B 주변에 하나만 ?있다면 확정시키기
    for k in direction:  # 상하좌우
        x2 = x1 + k[0]
        y2 = y1 + k[1]
        for l in may_p:
            if explore[x2][y2] == l and f == False:
                f = (x2, y2)
            elif explore[x2][y2] == l:
                f2 = True
    if f and (not f2):
        explore[f[0]][f[1]] = ' P '  # 확정!
        exscreen[f[0]][f[1]] = ' P '#확정!
        Px = f[0]
        Py = f[1]


# bfs일반화
def bfs(start, target, symbol):
    global ag, direction
    visited = np.zeros((6, 6))
    q = []
    who = {(start[0], start[1]): None}  # 왔던곳
    q.append((start[0], start[1]))
    visited[start[0]][start[1]] = True
    global canKill
    flag = True
    while q and flag:
        a = q.pop(0)
        for i in reversed(direction):  # 상하좌우
            x = a[0] + i[0]
            y = a[1] + i[1]
            if safe_zone[x][y] == 1 and visited[x][y] == False:  # 방문하지 않았고 갈수있는 곳이라면
                visited[x][y] = True
                q.append((x, y))
                who[(x, y)] = ((a[0], a[1]))
            if symbol:
                if symbol == 'WP?': #suicide에서 쓰임 W?나 P?나 WP?를 찾아서 간다!
                    b = explore[x][y]
                    if b==' W?' or b==' P?' or b=='WP?':
                        target = (10, 10)
                        who[(x, y)] = (a[0], a[1])
                        who[(10, 10)] = (x, y)
                        flag = False
                        break

                elif canKill[x][y]:  # Symbol을 찾으면 멈춤
                    target = (10, 10)
                    who[(x, y)] = (a[0], a[1])
                    who[(10, 10)] = (x, y)
                    flag = False
                    break
            elif target:
                if x == target[0] and y == target[1]:  # 목표지점에 도달했으면 멈춤
                    who[(x, y)] = ((a[0], a[1]))
                    flag = False
                    break
    trace = []
    top = (target[0], target[1])
    while top != (start[0], start[1]):
        trace.append(who[top])
        top = who[top]
    trace.reverse()
    if trace:
        return trace[1:]
    
def trace_check(trace):
    global preloc, ag, outt
    b=ag._curLoc
    if not check[b[0]][b[1]]:
        check[b[0]][b[1]]=1
        percept = percept_percept(b)  # percept 받아오기
        reasoning(b, percept)  # reasoning 하기
        print_agent(b, percept)  # 현재 에이전트 상태,위치 출력
    if trace:
        for i in range(len(trace)):
            a=trace[i]
            if not check[a[0]][a[1]]:
                check[a[0]][a[1]]= 1
                percept = percept_percept(a)  # percept 받아오기
                reasoning(a, percept)  # reasoning 하기
                print_agent(a, percept)  # 현재 에이전트 상태,위치 출력
            else: 
                #print(a)
                outt.append([a])
        preloc = trace[-1]
        ag._curLoc = trace[-1]

preloc = (1, 1)
def dfs_world(loc):
    global Gold_flag, percept, check, preloc, ag
    if not check[abs(loc[0])][abs(loc[1])]:
        check[abs(loc[0])][abs(loc[1])] = 1
        if loc[0] <= 0:
            percept = percept_percept((-loc[0], -loc[1]))
        else:
            percept = percept_percept(loc)  # percept 받아오기
        if not Gold_flag and ag._isAlive:  # 골드를 찾지 못하고 살아있을 때(무조건 살아있음)
            trace2 = bfs(preloc, (abs(loc[0]), abs(loc[1])), '')  # 다음 노드로 가는 경로 출력(bfs)      
            trace_check(trace2)
            if loc[0] <= 0:  # 벽에부딪혔을때
                print_agent(loc, percept)
                if trace2: preloc = trace2[-1]
            else:
                ag._curLoc = loc
                reasoning(loc, percept)  # reasoning 하기
                print_agent(loc, percept)  # 현재 에이전트 상태,위치 출력
                preloc = loc
                while safe_q:
                    safe_area = safe_q.pop()
                    dfs_world(safe_area)


def findSymbol(Symbol, Symbol2):
    global ag
    flag = False  # Symbol이 하나라도 존재하는지
    canKill = np.zeros((6, 6))
    whereWum = [[None for j in range(6)] for i in range(6)]
    for i in range(6):
        for j in range(6): whereWum[i][j] = []
    for i in range(1, 5):  # W or W?(Symbol)찾아서 룩형태 canKill에 1넣어주기
        for j in range(1, 5):
            a = explore[i][j]
            if a == Symbol or a == Symbol2:
                flag = True
                for k in range(1, 5):
                    if j != k:
                        canKill[i][k] = 1
                        whereWum[i][k].append((i, j))
                for k in range(1, 5):
                    if i != k:
                        canKill[k][j] = 1
                        whereWum[k][j].append((i, j))
    return flag, canKill, whereWum
# W? or P? 가서 죽어보기 안죽으면 거기서 탐색
# 죽었다면 완료
# 모든 W? or P를 탐색후 살아있으면
#
# 그냥 죽기
def suicide():
    global preloc, ag, outt
    while ag._isAlive:
        flag2 = False
        for i in range(1, 5):  # WP?가 있으면 탐색하며 죽기
            for j in range(1, 5):
                if explore[i][j] == ' W?' or explore[i][j] == ' P?' or explore[i][j] == 'WP?':
                    flag2 = True
                    break
            if flag2: break
        if flag2:
            trace = bfs(ag._curLoc, None, 'WP?') #현재위치에서 W?나 P?를 찾아간다
            if trace:
                trace_check(trace[:-1])
                dfs_world(trace[-1])#WP?있는곳 탐색
        else: 
            ag._isAlive = False
            return False
            break #WP?가 없으니까 나가서 그냥죽기
    #print((1,1))
    outt.append([(1,1)])
    preloc=(1,1)
    ag = Agent() 
    return True

def wumKill(wumloc1):
    world[wumloc1[0]][wumloc1[1]] = ' O '
    for i in direction:
        x = wumloc1[0] + i[0]
        y = wumloc1[1] + i[1]
        if world[x][y] == ' S ' or world[x][y] == 'SB ':
            world[x][y] = ' O '
    for j in range(1, 5):  # SB다시 만들어주기
        for k in range(1, 5):
            if world[j][k] == ' W ':
                for i in direction:
                    x = j + i[0]
                    y = k + i[1]
                    xy = world[x][y]
                    if xy != ' W ' and xy != ' P ' and xy != ' G ' and xy != ' X ':
                        if xy == ' B ': 
                            world[x][y] = 'SB '
                        else: 
                            world[x][y] = ' S '
            elif world[j][k] == ' P ':
                for i in direction:
                    x = j + i[0]
                    y = k + i[1]
                    xy = world[x][y]
                    if xy != ' W ' and xy != ' P ' and xy != ' G ' and xy != ' X ':
                        if xy == ' S ':
                            world[x][y] = 'SB '
                        else:
                            world[x][y] = ' B '

def count_head(result):
    action_count = 0
    if result == 'a':
        action_count += 1
    elif result == 'b':
        action_count += 2
    elif result == 'c':
        action_count += 3
    elif result == 'd':
        action_count += 2
    return action_count

def mv_act (posy,posx,nposy,nposx) :
  global oUI
  count = 0
  if posy == 10: return 0
  dify = nposy - posy
  difx = nposx - posx
  mv_cnt = 0
  global ag
  if (ag._curDir ==0):
    if(dify ==0 and difx >=1):
      #print ('GoForward')
      oUI.append('GoForward')
      result = 'a'
      ag._curDir =0
      mv_cnt +=1
      count = count_head(result)
    elif(dify >= 1 and difx ==0):
      #print ('TurnLeft GoForward')
      oUI.append('TurnLeft GoForward')
      result = 'b'
      ag._curDir =1
      mv_cnt +=2
      count = count_head(result)
    elif(dify == 0 and difx <=-1):
      #print ('TurnLeft TurnLeft GoForward')
      oUI.append('TurnLeft TurnLeft GoForward')
      ag._curDir =2
      result = 'c'
      mv_cnt +=3
      count = count_head(result)
    elif(dify <=-1 and difx == 0):
      #print('TurnRight GoForward')
      oUI.append('TurnRight GoForward')
      result ='d'
      ag._curDir =3
      mv_cnt +=2
      count = count_head(result)

    
  elif (ag._curDir ==1):
      if(dify ==0 and difx >=1):
        result = 'd'
        #print('TurnRight GoForward')
        oUI.append('TurnRight GoForward')
        ag._curDir =0
        count = count_head(result)
      elif(dify >= 1 and difx ==0):
        result ='a'
        #print ('GoForward')
        oUI.append('GoForward')
        ag._curDir =1
        count = count_head(result)

      elif(dify == 0 and difx <=-1):
         result = 'b'
         #print ('TurnLeft GoForward')
         oUI.append('TurnLeft GoForward')
         ag._curDir =2
         count = count_head(result)
         
      elif(dify <=-1 and difx == 0):
        #print ('TurnLeft TurnLeft GoForward')
        oUI.append('TurnLeft TurnLeft GoForward')
        result = 'c'
        ag._curDir=3
        mv_cnt +=3
        count = count_head(result)


  elif (ag._curDir == 2):
      if(dify ==0 and difx >=1):
        oUI.append('TurnLeft TurnLeft GoForward')
        #print ('TurnLeft TurnLeft GoForward')
        result = 'c'
        ag._curDir = 0
        mv_cnt +=3
        count = count_head(result)
      elif(dify >= 1 and difx ==0):
        oUI.append('TurnRight GoForward')
        #print('TurnRight GoForward')
        result = 'd'
        ag._curDir = 1
        mv_cnt +=2
        count = count_head(result)
        
      elif(dify == 0 and difx <=-1):
        oUI.append('GoForward')
        #print ('GoForward')
        result = 'a'
        ag._curDir =2
        mv_cnt +=1
        count = count_head(result)

      elif(dify <=-1 and difx == 0):
        oUI.append('TurnLeft GoForward')
        #print ('TurnLeft GoForward')
        result = 'b'
        ag._curDir =3
        mv_cnt +=2
        count = count_head(result)
        
  elif (ag._curDir == 3):
      if(dify ==0 and difx >=1):
        oUI.append('TurnLeft GoForward')
        #print ('TurnLeft GoForward')
        result = 'b'
        ag._curDir =0
        mv_cnt +=2
        count = count_head(result)
      elif(dify >= 1 and difx ==0):
        oUI.append('TurnLeft TurnLeft GoForward')
        #print ('TurnLeft TurnLeft GoForward')
        result = 'c'
        ag._curDir =1
        mv_cnt +=3
        count = count_head(result)
      elif(dify == 0 and difx <=-1):
        oUI.append('TurnRight GoForward')
        #print('TurnRight GoForward')
        result = 'd'
        ag._curDir =2
        mv_cnt +=2
        count = count_head(result)
      elif(dify <=-1 and difx == 0):
        oUI.append('GoForward')
        #print ('GoForward')
        result = 'a'
        ag._curDir = 3
        mv_cnt +=1
        count = count_head(result)
  return count 

def mv_act2(posy, posx, nposy, nposx):
    count =0
    dify = nposy - posy
    difx = nposx - posx
    mv_cnt = 0
    global ag
    if (ag._curDir == 0):
        if (dify == 0 and difx >= 1):
            # print ('GoForward')
            result = 'a'
            ag._curDir = 0
            mv_cnt += 1
            count = count_head(result)
        elif (dify >= 1 and difx == 0):
            # print ('TurnLeft GoForward')
            result = 'b'
            ag._curDir = 1
            mv_cnt += 2
            count = count_head(result)
        elif (dify == 0 and difx <= -1):
            # print ('TurnLeft TurnLeft GoForward')
            ag._curDir = 2
            result = 'c'
            mv_cnt += 3
            count = count_head(result)
        elif (dify <= -1 and difx == 0):
            # print('TurnRight GoForward')
            result = 'd'
            ag._curDir = 3
            mv_cnt += 2
            count = count_head(result)


    elif (ag._curDir == 1):
        if (dify == 0 and difx >= 1):
            result = 'd'
            # print('TurnRight GoForward')
            ag._curDir = 0
            count = count_head(result)
        elif (dify >= 1 and difx == 0):
            result = 'a'
            # print ('GoForward')
            ag._curDir = 1
            count = count_head(result)

        elif (dify == 0 and difx <= -1):
            result = 'b'
            # print ('TurnLeft GoForward')
            ag._curDir = 2
            count = count_head(result)

        elif (dify <= -1 and difx == 0):
            # print ('TurnLeft TurnLeft GoForward')
            result = 'c'
            ag._curDir = 3
            mv_cnt += 3
            count = count_head(result)


    elif (ag._curDir == 2):
        if (dify == 0 and difx >= 1):
            # print ('TurnLeft TurnLeft GoForward')
            result = 'c'
            ag._curDir = 0
            mv_cnt += 3
            count = count_head(result)
        elif (dify >= 1 and difx == 0):
            # print('TurnRight GoForward')
            result = 'd'
            ag._curDir = 1
            mv_cnt += 2
            count = count_head(result)

        elif (dify == 0 and difx <= -1):
            # print ('GoForward')
            result = 'a'
            ag._curDir = 2
            mv_cnt += 1
            count = count_head(result)

        elif (dify <= -1 and difx == 0):
            # print ('TurnLeft GoForward')
            result = 'b'
            ag._curDir = 3
            mv_cnt += 2
            count = count_head(result)

    elif (ag._curDir == 3):
        if (dify == 0 and difx >= 1):
            # print ('TurnLeft GoForward')
            result = 'b'
            ag._curDir = 0
            mv_cnt += 2
            count = count_head(result)
        elif (dify >= 1 and difx == 0):
            # print ('TurnLeft TurnLeft GoForward')
            result = 'c'
            ag._curDir = 1
            mv_cnt += 3
            count = count_head(result)
        elif (dify == 0 and difx <= -1):
            # print('TurnRight GoForward')
            result = 'd'
            ag._curDir = 2
            mv_cnt += 2
            count = count_head(result)
        elif (dify <= -1 and difx == 0):
            # print ('GoForward')
            result = 'a'
            ag._curDir = 3
            mv_cnt += 1
            count = count_head(result)
    return count

def mv_act3(posy, posx, nposy, nposx):
    dify = nposy - posy
    difx = nposx - posx
    mv_cnt = 0
    global ag
    count =0
    if (ag._curDir == 0):
        if (dify == 0 and difx >= 1):
            outt.append(["No Rotate"])
            #print('No Rotate')
            result = 'a'
            ag._curDir = 0
            mv_cnt += 1
            count = count_head(result)
        elif (dify >= 1 and difx == 0):
            outt.append(["TurnLeft"])
            #print('TurnLeft')
            result = 'b'
            ag._curDir = 1
            mv_cnt += 2
            count = count_head(result)
        elif (dify == 0 and difx <= -1):
            outt.append(["TurnLeft TurnLeft"])
            #print('TurnLeft TurnLeft')
            ag._curDir = 2
            result = 'c'
            mv_cnt += 3
            count = count_head(result)
        elif (dify <= -1 and difx == 0):
            outt.append(["TurnRight"])
            #print('TurnRight')
            result = 'd'
            ag._curDir = 3
            mv_cnt += 2
            count = count_head(result)


    elif (ag._curDir == 1):
        if (dify == 0 and difx >= 1):
            result = 'd'
            outt.append(["TurnRight"])
            #print('TurnRight')
            ag._curDir = 0
            count = count_head(result)
        elif (dify >= 1 and difx == 0):
            result = 'a'
            outt.append(["No Rotate"])
            #print('No Rotate')
            ag._curDir = 1
            count = count_head(result)

        elif (dify == 0 and difx <= -1):
            result = 'b'
            outt.append(["TurnLeft"])
            #print('TurnLeft')
            ag._curDir = 2
            count = count_head(result)

        elif (dify <= -1 and difx == 0):
            outt.append(["TurnLeft TurnLeft"])
            #print('TurnLeft TurnLeft')
            result = 'c'
            ag._curDir = 3
            mv_cnt += 3
            count = count_head(result)


    elif (ag._curDir == 2):
        if (dify == 0 and difx >= 1):
            outt.append(["TurnLeft TurnLeft"])
            #print('TurnLeft TurnLeft')
            result = 'c'
            ag._curDir = 0
            mv_cnt += 3
            count = count_head(result)
        elif (dify >= 1 and difx == 0):
            outt.append(["TurnRight"])
            #print('TurnRight')
            result = 'd'
            ag._curDir = 1
            mv_cnt += 2
            count = count_head(result)

        elif (dify == 0 and difx <= -1):
            outt.append(["No Rotate"])
            #print('No Rotate')
            result = 'a'
            ag._curDir = 2
            mv_cnt += 1
            count = count_head(result)

        elif (dify <= -1 and difx == 0):
            outt.append(["TurnLeft"])
            #print('TurnLeft')
            result = 'b'
            ag._curDir = 3
            mv_cnt += 2
            count = count_head(result)

    elif (ag._curDir == 3):
        if (dify == 0 and difx >= 1):
            outt.append(["TurnLeft"])
            #print('TurnLeft')
            result = 'b'
            ag._curDir = 0
            mv_cnt += 2
            count = count_head(result)
        elif (dify >= 1 and difx == 0):
            outt.append(["TurnLeft TurnLeft"])
            #print('TurnLeft TurnLeft')
            result = 'c'
            ag._curDir = 1
            mv_cnt += 3
            count = count_head(result)
        elif (dify == 0 and difx <= -1):
            outt.append(["TurnRight"])
            #print('TurnRight')
            result = 'd'
            ag._curDir = 2
            mv_cnt += 2
            count = count_head(result)
        elif (dify <= -1 and difx == 0):
            outt.append(["No Rotate"])
            #print('No Rotate')
            result = 'a'
            ag._curDir = 3
            mv_cnt += 1
            count = count_head(result)
    return count

# 123
def W_Arrow():
    global canKill, preloc, ag, outt
    isWum = True
    while isWum:
        FindW = findSymbol(' W ', ' W ')  # 1 W찾아 룩형태 가서 화살쏘기
        if FindW[0]:  # W가 하나라도 존재할때
            canKill = FindW[1]
            whereWum = FindW[2]
            trace4 = []
            if canKill[ag._curLoc[0]][ag._curLoc[1]]:
                trace4.append(ag._curLoc)
            else:
                trace4 = bfs(ag._curLoc, None, ' W ')  # 가기
            if trace4:
                if len(trace4)!=1:
                    trace_check(trace4)
                loc = trace4[-1]
                wumloc = whereWum[loc[0]][loc[1]]
                cnt = mv_act2(loc[0], loc[1], wumloc[0][0], wumloc[0][1])
                minx, miny = wumloc[0][0], wumloc[0][1]
                for i in range(len(wumloc) - 1):  # wum갯수만큼
                    x = wumloc[i][0]
                    y = wumloc[i][1]
                    pcnt = mv_act2(loc[0], loc[1], x, y)
                    if cnt > pcnt:
                        cnt = pcnt
                        minx, miny = x, y
                preloc = loc
                wumloc1 = (minx, miny)
                if ag._arrowCnt > 0:
                    mv_act3(loc[0], loc[1], wumloc1[0], wumloc1[1])  # 방향전환 출력
                    ag._arrowCnt -= 1                   
                    #print("Shoot arrow from ", end='')
                    #print(preloc, end='')
                    #print(" to ", end='')
                    #print(wumloc1)
                    #print([' None ', ' None ', ' None  ', 'None', 'Scream'])
                    #print()
                    outt.append(["Shoot arrow from " + str(preloc) + " to "+str(wumloc1),[' None ', ' None ', ' None  ', 'None', 'Scream']])
                    wumKill(wumloc1)
                    check[wumloc1[0]][wumloc1[1]] = 0
                    for j in direction:  # 죽은곳 상하좌우
                        x = wumloc1[0] + j[0]
                        y = wumloc1[1] + j[1]
                        check[x][y] = 0
                    dfs_world((wumloc1[0], wumloc1[1]))  # Wumpus가 죽은곳 탐색
                    W_Arrow()
                    Ww_Arrow()
                else:
                    #print("화살이 부족합니다")
                    outt.append(["lack of arrows"])
                    a= suicide()
                    if not a:
                        #print("suicide")
                        outt.append(["suicide", (1, 1)])
                        #print((1,1))
                        preloc=(1,1)
                        ag = Agent()                    
                    W_Arrow()
                    Ww_Arrow()
        else:
            isWum = False

def Ww_Arrow():
    isWum = True
    global canKill, preloc, ag, outt
    while isWum:      
        FindWw = findSymbol(' W?', 'WP?')  # W? 화살 쏴보기
        if FindWw[0]:
            canKill = FindWw[1]
            whereWum = FindWw[2]
            trace5 = []
            if canKill[ag._curLoc[0]][ag._curLoc[1]]:
                trace5.append(ag._curLoc)
            else:
                trace5 = bfs(ag._curLoc, None, ' W?')  # 가기
            if trace5:  
                if len(trace5)!=1:
                    trace_check(trace5)
                loc = trace5[-1]
                wumloc = whereWum[loc[0]][loc[1]]
                cnt = mv_act2(loc[0], loc[1], wumloc[0][0], wumloc[0][1])
                minx, miny = wumloc[0][0], wumloc[0][1]
                for i in range(len(wumloc) - 1):  # wum갯수만큼
                    x = wumloc[i][0]
                    y = wumloc[i][1]
                    pcnt = mv_act2(loc[0], loc[1], x, y)
                    if cnt > pcnt:
                        cnt = pcnt
                        minx, miny = x, y
                preloc = loc
                wumloc1 = (minx, miny)
                if ag._arrowCnt > 0:
                    mv_act3(loc[0], loc[1], wumloc1[0], wumloc1[1])  # 방향전환 출력
                    ag._arrowCnt -= 1
                    # print("Shoot arrow! from ", end='')
                    # print(preloc, end='')
                    # print(" to ", end='')
                    # print(wumloc1)
                    outt.append(["Shoot arrow from "+ str(preloc)+ " to "+str(wumloc1)])
                    if world[minx][miny]==' W ':
                        #print([' None ', ' None ', ' None  ', 'None', 'Scream'])
                        outt.append([[' None ', ' None ', ' None  ', 'None', 'Scream']])
                    else: 
                        #print([' None ', ' None ', ' None  ', 'None', ' None'])
                        outt.append([[' None ', ' None ', ' None  ', 'None', ' None']])
                    #print()
                    wumKill(wumloc1)
                    check[wumloc1[0]][wumloc1[1]] = 0
                    for j in direction:  # 죽은곳 상하좌우
                        x = wumloc1[0] + j[0]
                        y = wumloc1[1] + j[1]
                        check[x][y] = 0
                    dfs_world((wumloc1[0], wumloc1[1]))  # Wumpus가 죽은곳 탐색
                    W_Arrow()# 다시W찾아보기
                    Ww_Arrow()
                else:
                    a=suicide() # W? or P?인 곳 몸으로 탐색하며 죽기 없을때는 그냥 죽기
                    if not a:
                        # print("suicide")  
                        # print((1,1))
                        outt.append(["suicide",(1,1)])
                        preloc=(1,1)
                        ag = Agent() 
                    W_Arrow()
                    Ww_Arrow()
        else:
            isWum = False


oUI =deque()
oUI2 =deque()
def print_oUI():
    global oUI
    # GUI구현
    tk = Tk()
    tk.title('Wumpus_World')
    tk.geometry('1280x800')
    #tk.resizable(False,False)
    # 함수 정의 (버튼을 누르면 텍스트 내용이 바뀜)
    def event():
        pass
    explore3 = np.array([[' X ', ' X ', ' X ', ' X ', ' X ', ' X '],
                [' X ', ' ? ', ' ? ', ' ? ', ' ? ', ' X '],
                [' X ', ' ? ', ' ? ', ' ? ', ' ? ', ' X '],
                [' X ', ' ? ', ' ? ', ' ? ', ' ? ', ' X '],
                [' X ', ' ? ', ' ? ', ' ? ', ' ? ', ' X '],
                [' X ', ' X ', ' X ', ' X ', ' X ', ' X ']])  # 탐험중정보
    
    frame=Frame(tk)
    frame.pack(side=RIGHT)
    scrollbar=Scrollbar(frame)
    scrollbar.pack(side=RIGHT)
    button2=Listbox(frame, yscrollcommand=scrollbar.set,font=('Consolas',18),width=55,height=30)


    button0 = Button(tk,text=oUI[0],font=('Consolas',18))
    button1 = Button(tk,text=oUI[1],font=('Consolas',18),command=event)
    button4 = Button(tk,text='State History:',font=('Consolas',18))
    #button2 = Button(tk,text='NONE',font=('Consolas',18),width=55,height=30)
    button3 = Button(tk,text=explore3,font=('Consolas',18),command=event)
    #button4 = Button(tk,text='NEXT',font=('Consolas',13))
    button0.place(x=10,y=10)
    button1.place(x=10,y=60)
    button4.place(x=10,y=250)
    button3.place(x=10,y=300)
    button2.pack(side=LEFT)
    scrollbar.config(command=button2.yview)

    #button4.pack(side=RIGHT)

    for i in range(2,len(oUI)):
        dt=str(type(oUI[i]))
        a=oUI[i]
        if dt=="<class 'numpy.ndarray'>" and ' X ' in a:
            button3['text'] =a
            button2.delete(0,END)
            tk.update()
            time.sleep(1)
            
        else:
            button2.insert(END, '\n')
            button2.insert(END, str(a))
            tk.update()
            time.sleep(1)
            #button2['text'] +='\n'
            #button2['text'] +=str(a) 
    tk.mainloop()

def print_outt(outt):
    global ag,oUI
    ploc = (1,1)
    flg=True
    for i in range(len(outt)):
        if flg:
            for j in range(len(outt[i])):
                if flg:
                    dt=str(type(outt[i][j]))
                    a=outt[i][j]
                    if dt=="<class 'str'>":
                        print(a)
                        oUI.append(a)
                        if a=='Climb':
                            flg=False

                        elif a=='Dead' or a=='suicide':
                            ag=Agent()
                            ploc= (1,1)
                    elif dt=="<class 'tuple'>":
                        if not a[0] <=0:      
                            print(a)
                            oUI.append(a)                
                            mv_act(ploc[0],ploc[1],a[0],a[1])              
                            ploc = a
                        else:                    
                            print((abs(a[0]),abs(a[1])))
                            oUI.append((abs(a[0]),abs(a[1])))
                            mv_act(ploc[0],ploc[1],abs(a[0]),abs(a[1]))            
                    elif dt=="<class 'numpy.ndarray'>":
                        oUI.append(a)
                        #oUI.append('')
                        print(a)
                        print()
                    else : 
                        oUI.append(a)
                        print(a)

    print_oUI()



flag5 = False
while not flag5:
    world = np.array([[' X ', ' X ', ' X ', ' X ', ' X ', ' X '],
                  [' X ', ' O ', ' O ', ' O ', ' O ', ' X '],
                  [' X ', ' O ', ' O ', ' O ', ' O ', ' X '],
                  [' X ', ' O ', ' O ', ' O ', ' O ', ' X '],
                  [' X ', ' O ', ' O ', ' O ', ' O ', ' X '],
                  [' X ', ' X ', ' X ', ' X ', ' X ', ' X ']])  # 실제월드
    init_World()
    if world[1][1]==' O ':
        flag5 = True

#print("World INIT... The World is")
outt.append(["World INIT... The World is"])
#print(np.flip(world, axis=-2))
world3=copy.deepcopy(world)
outt.append([np.flip(world3, axis=-2)])
print()
# 이 밑부터 Main함수
ag = Agent()
explore[1][1] = ' O '
exscreen[1][1] = ' O '


dfs_world((1, 1))
W_Arrow()
Ww_Arrow()
sflag = True
while sflag:
    #suicide만하면됌
    sflag = suicide()
    W_Arrow()
    Ww_Arrow()
    if not sflag:
        trace = bfs(ag._curLoc, (1,1), None)
        if trace:
            trace.append((1,1))
            for i in range(len(trace)):
                #print(trace[i])
                pass
            outt.append(trace)    

#print("Climb")
#print("탐색실패")
percept = np.array([' None ', ' None ', ' None  ', 'None', ' None'])
print_agent((1,1), percept)
outt.append(["search failed", "Climb" ])

print_outt(outt)



