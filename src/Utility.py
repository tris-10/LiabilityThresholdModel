# -------------------------------------------------------------------------------
# Name:        Utility.py
# Purpose: Some stand alone methods doing a variety of things, may want to break this up into
# different functionality groups in the future like: IO, Logging, etc
#
# Author:      Tristan Hayeck
#
# Created:     12/2015
# Copyright:   (c) Hayeck 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import argparse,math, sys, os,random, datetime, subprocess, re
import numpy as np
import pandas as pd
from scipy import linalg
from scipy import optimize
from scipy import stats

#from itertools import izip_longest

import MatrixUtils


class MyRuntimeException(Exception):
    """
    Class for Runtime exceptions which we handle,
    and print a friendly error message for,
    without python traceback.
    """


def PROGRESS(msg, printit=True, printTime=True):
    if printit:
        sys.stderr.write("%s\n"%msg.strip())
        if printTime:
            print datetime.datetime.utcnow()


def removeWhiteSpaceFromString(theString):
    return re.sub(r'\s+','',theString)


def runJobsInParallel(commands,threads, DEBUG):
    """
    This will run the set of jobs with the maximum number of threads specified
    :param commands: The commands to be run. This is a list of lists, the outer list
    are all of the commands and the inner list consists of the arguments within a command
    :param threads: Maximum allowable threads to be used
    :param DEBUG: detailed debug mode
    :return:
    """
    numJobs = len(commands)
    numSubSets =  int(math.ceil(float(numJobs) / threads))
    PROGRESS("About to run jobs", DEBUG)
    print "Will be running %s jobs" % numJobs
    for i in range(numSubSets):
        children = []
        subSetCommands =[]
        if (i+1)*threads < numJobs:
            subSetCommands = commands[i*threads: (i+1)*threads]
        else:
            subSetCommands = commands[i*threads: numJobs]
        #parallelizing the jobs into subsets
        for cmd in subSetCommands:
            if DEBUG:
                print "the command is: ",cmd
            p = subprocess.Popen(cmd)
            if p:
                children.append(p)
            else:
                print "Throw exception"
        n = len(children)
        if DEBUG:
            for j in range(n):
                os.waitpid(-1, 0)
                if j % 10 == 0:
                    PROGRESS( "for subset %s finished %s out of %s jobs, or %s jobs out of a total of %s" % \
                          (i, j, n,  i*threads+j,numJobs))



def splitFile(fileName,linesPerFile,smallFilePrefix, includeHeader=True, DEBUG=True):
    """
    Breaks up a large file into smaller files

    modified from
    http://stackoverflow.com/questions/16289859/splitting-large-text-file-into-smaller-text-files-by-line-numbers-using-python
    possibly better ways of doing this, just a quick fix
    :param fileName: The file to be broken up
    :param linesPerFile: number of lines per file (excluding the header if it's included)
    :param smallFilePrefix: name of the smaller file smallFilePrefix[id].txt
    :param includeHeader: print the header line in all the files
    :param DEBUG: print out details as we go
    :return:
    """
    curFile = None
    fileCounter = 0
    with open(fileName) as bigfile:
        for lineNumber, line in enumerate(bigfile):
            if lineNumber == 0:
                headerString = line
            #first file we want the header + lines per file
            linesPerFileAdjusted = linesPerFile + int( includeHeader and fileCounter == 0)
            adjustedLineNum = lineNumber - int(includeHeader)

            if adjustedLineNum % linesPerFile == 0:
                if curFile:
                    curFile.close()
                    fileCounter+=1
                curFileName = smallFilePrefix+str(fileCounter)+".txt"

                curFile = open(curFileName, "w")
                PROGRESS("Opening breaking up big file %s into %s"% (fileName, curFileName)\
                         ,DEBUG,DEBUG)
                if includeHeader: #and not fileCounter == 0
                    curFile.write(headerString)
            if adjustedLineNum > -1:
                curFile.write(line)
        if curFile:
            curFile.close()
            fileCounter+=1


# print "Command length: "
            # print len(commands[0])
            # print "current command: "
            # print cmd
            # print "Command length: "
            # print len(commands[0])
            # print "current command: "
            # print cmd

def skipStartingLines(inputFileName,outputFileName, linesToSkip):
    """
    Output a file skipping the first set of lines.
    :param inputFileName:
    :param outputFileName:
    :param linesToSkip:
    :return:
    """
    inputFile = open(inputFileName,"r")
    outputFile = open(outputFileName,"w")
    lineCount = 0
    for line in inputFile:
        if lineCount > linesToSkip:
            outputFile.write(line)
        lineCount +=1
    inputFile.close()
    outputFile.close()