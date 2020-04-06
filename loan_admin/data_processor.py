class DataProcessor:
    def __init__(self):
        pass

    def process(read, write):
        fw = open(write, 'w')
        fr = open(read, 'r')
        line = fr.readline()
        count = 0
        while(line != ""):
            flag = 0
            lout = ""
            for i in range(0, len(line)):
                if(flag == 1):
                    lout = lout + line[i]
                if(line[i] == ','):
                    flag = 1
            fw.write(lout)
            line = fr.readline()
            count = count + 1
        fw.close()
        fr.close()
