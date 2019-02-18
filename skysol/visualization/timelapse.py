#!/usr/bin/env /sat/san/Programme/python/python3.3/bin/python3

"""
title        : timelapse.py
description  : Script to generate timelapse movies from subsequent camera images using "Mencoder"
author       : Thomas Schmidt
setup date   : 20150821
last change  : 20150915
version      : 0.1
usage        : python timelapse.py
history      : 20150821: setup
	       20150915: send emails only if not succesful to avoid spamming
python_version    :3.3.5
==============================================================================
"""
import subprocess                 # For issuing commands to the OS.
import sys
import getopt
from datetime import datetime
import glob
import os
import shutil
# drawing modules
from PIL import Image, ImageDraw, ImageFont
# E-mail modules
import smtplib
from email.mime.text import MIMEText


def to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError



# define defaults
params = {}
params['picpath'] = '.' # Path for input images
params['format'] = "png" # Input image format
params['vidpath'] = '.' # Path for output video
params['vfname'] = 'timelapse.avi' # Video filename
params['fps'] = "30" # Frames per second
params['scale'] = "768:768" # resolution
params['indate'] = None # Date format:YYYYMMDD
params['namefmt'] = "%Y%m%d_%H%M%S" # the filename format of each image
params['draw_datetime'] = "False" # The date and time of each image written in the image
params['draw_label'] = None # A label written in all images
params['send_mail'] = None # E-mail address
params['check'] = "True" # to check before if mencoder exists
params['regexp'] = None # regular expression for input images, replaces picpath and date
params['clean'] = "False" # if true, images will be deleted afterwards

USAGE = "USAGE:\n" \
    "python timelapse.py\n\n" \
    "\t--picpath=\t string, image directory\n"\
    "\t--vidpath=\t string, output directory for video\n"\
    "\t--format=\t string, image format(ending), default png\n"\
    "\t--date=\t string, Input date (format YYYYMMDD), is added to picpath as subdirectory, default is '"+str(params['indate'])+"' +\n"\
    "\t--namefmt=\t string,format of filenames, default '"+str(params['namefmt'])+"'\n"\
    "\t--vfname=\t string, video filename, default '"+str(params['vfname'])+"'\n" \
    "\t--draw_label=\t string, label which is drawn in the corner, default is empty\n" \
    "\t--draw_datetime=\t boolean, draws Date and Time in the corner, default "+str(params['draw_datetime'])+"\n" \
    "\t--send_mail=\t string, mail address where a final mail is send, useful for operational purposes, default "+str(params['send_mail'])+"\n" \
    "\t--fps=\t string, frames per second (fps) used, default "+params['fps']+"\n" \
    "\t--scale=\t string, horizontal and vertical size for video frames, default is "+params['scale']+"\n" \
    "\t--check=\t boolean, check before executing if mencoder exists, default is "+str(params['check'])+"\n" \
    "\t--regexp=\t string, regular expression for input image filenames, if used, it replaces picpath, date, and namefmt. default is "+str(params['regexp'])+"\n" \
    "\t--clean=\t boolean, deletes all images after video creation. default is "+str(params['clean'])+"\n"

USAGE = USAGE + """
    \nExample: timelapse.py --draw_datetime=True --draw_label="Wechloy"
    --picpath=/sat/san/messungen/skyimager --format=jpg --check=False
    --send_mail="t.schmidt@uni-oldenburg.de" --date=20150823
        """

# read arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], "h", \
    ["picpath=", "regexp=", "format=", "vidpath=", "vfname=", "fps=", "scale=", \
     "date=", "namefmt=", "draw_datetime=", "draw_label=", "send_mail=", \
     "check=", "clean="])
except getopt.GetoptError as er:
    print("\n" + str(er) + "\n\n" + USAGE)
    sys.exit(2)




for opt, arg in opts:
    if opt == '-h':
        print(USAGE)
        sys.exit()
    elif opt in ("--date"):
        params['indate'] = str(arg)
    elif opt in ("--picpath"):
        params['picpath'] = str(arg)
    elif opt in ("--regexp"):
        params['regexp'] = str(arg)
    elif opt in ("--format"):
        params['format'] = str(arg)
    elif opt in ("--vidpath"):
        params['vidpath'] = str(arg)
    elif opt in ("--vfname"):
        params['vfname'] = str(arg)
    elif opt in ("--fps"):
        params['fps'] = str(arg)
    elif opt in ("--fps"):
        params['scale'] = str(arg)
    elif opt in ("--scale"):
        params['scale'] = str(arg)
    elif opt in ("--namefmt"):
        params['namefmt'] = str(arg)
    elif opt in ("--draw_datetime"):
        params['draw_datetime'] = arg
    elif opt in ("--draw_label"):
        params['draw_label'] = arg
    elif opt in ("--check"):
        params['check'] = arg
    elif opt in ("--send_mail"):
        params['send_mail'] = arg
    elif opt in ("--clean"):
        params['clean'] = arg

# argument processing
if params['send_mail'] is not None:
    stdout  = sys.stdout
    sys.stdout = open('/tmp/timelapse.log','w')


if params['indate'] is not None:
    actdate = datetime.strptime(params['indate'],"%Y%m%d")
    params['picpath'] = params['picpath'] + '/' + params['indate']

print(params['regexp'])
if params['regexp'] is not None:
    params['picpath'] = params['regexp']

tmpdir = '/tmp/pics'


#----------------------
# Image preprocessing
#----------------------
if (to_bool(params['draw_datetime']) is True) or (params['draw_label'] is not None):
    print('\nProcessing images from ' + params['picpath'] + '/' + '*.' + params['format'] + \
          ' and buffering them to '+tmpdir+'\n')
    # create temporary directory for processed images
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    images = sorted(glob.glob(params['picpath'] + '/' + '*.' + params['format']))
    params['picpath'] = tmpdir
    for img in images:
        dtstring = img.split('/')[-1]
        try:
        	dt = datetime.strptime(dtstring,params['namefmt']+'.'+params['format'])
        except ValueError:
        	continue
        try:
            image = Image.open(img)
        except:
            pass
        draw = ImageDraw.Draw(image)
        lx, ly = image.size

        txtfont = ImageFont.truetype('/usr/share/fonts/liberation/LiberationSans-Regular.ttf', 50)

        # Draw Timestring
        string = dt.strftime("%H:%M:%S UTC")
        if dt: draw.text((lx-350, 20),string,fill = 'red',font=txtfont)

        # Draw Datestring
        string = dt.strftime("%Y/%m/%d")
        if dt: draw.text((20, 20),string,fill = 'red',font=txtfont)

        # Draw Location
        if params['draw_label']:
            string = params['draw_label']
            draw.text((20, ly-80),string,fill = 'red',font=txtfont)

        image.save(tmpdir + '/' + dtstring)

#-------------------------
# Mail settings
#-------------------------
if params['send_mail']:

    FROM = 'mess_meteo@srvlxmess01'
    TO = params['send_mail']
    SERVER = "localhost"
    SUBJECT = "Timelapse"
    server = smtplib.SMTP(SERVER)

# FROM = 'mess_meteo@srvlxmess01'
# TO = "t.schmidt@uni-oldenburg.de"
# SERVER = "localhost"
# SUBJECT = "Timelapse"
# server = smtplib.SMTP(SERVER)
# f = open('/tmp/timelapse.log','r')
# text = f.read()
# print(text)
# msg = MIMEText(text)
# msg['Subject'] = SUBJECT
# msg['From'] = FROM
# msg['To'] = TO
# server.sendmail(FROM, TO, msg.as_string())


if to_bool(params['check']) == True:
    not_found_msg = """
    The mencoder command was not found;
    mencoder is used by this script to make an avi file from a set of pngs.
    It is typically not installed by default on linux distros because of
    legal restrictions, but it is widely available.
    """
    print("Check mencoder availability\n----------------")
    try:
        subprocess.check_call(['mencoder'])
    except subprocess.CalledProcessError:
        print("mencoder command was found")
        pass # mencoder is found, but returns non-zero exit as expected
        # This is a quick and dirty check; it leaves some spurious output
        # for the user to puzzle over.
    except OSError:
        print(not_found_msg)
        sys.exit("quitting\n")
    print("-----------\nfinished")


# We want to use Python to make what would normally be a command line
# call to Mencoder.  Specifically, the command line call we want to
# emulate is (without the initial '#'):
# mencoder mf://*.png -mf type=png:w=800:h=600:fps=25 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o output.avi
# See the MPlayer and Mencoder documentation for details.
#

command = ('mencoder',
           'mf://'+params['picpath']+'/*.'+params['format'],
           '-mf',
           'type='+params['format']+':fps='+params['fps'],
           '-vf',
           'scale='+params['scale'],
           '-ovc',
           'x264',
           '-quiet',
           '-x264encopts',
           'crf=30:threads=auto:subq=7:frameref=5:bframes=2:nr=2000',
           '-o',
           params['vidpath']+'/'+params['vfname'])


#os.spawnvp(os.P_WAIT, 'mencoder', command)
print("\n\nabout to execute:\n%s\n\n" % ' '.join(command))
success=True
try:
    subprocess.check_call(command)
except subprocess.CalledProcessError as e:
    success=False

if success:
    print("\n\nThe movie was written to "+params['vidpath']+'/'+params['vfname'])
    if not params['clean']: print("\n\nYou may want to delete "+params['picpath']+"/"+'.'+params['format']+" now.\n\n")
    suc = ' successful'
else:
    print("\n\nError: The movie was NOT created")
    if not params['clean']:
        print ("\n\nYou may want to delete "+params['picpath']+"/"+'.'+params['format']+" now.\n\n")
    else:
        print("\n No images will be deleted.\n")
    suc = ' not successful'

if params['send_mail'] and not success:
    sys.stdout = stdout
    f = open('/tmp/timelapse.log','r')
    # send Mail
    text = f.read()
    print(text)
    msg = MIMEText(text)
    msg['Subject'] = SUBJECT + suc
    msg['From'] = FROM
    msg['To'] = TO
    server.sendmail(FROM, TO, msg.as_string())


# Clean up
if os.path.exists(tmpdir):
    shutil.rmtree(tmpdir)
#if params['clean'] and success:
#    for f in glob.glob(params['picpath'] + '/' + '*.' + params['format']):
#        os.remove(f)

# Close mail
if params['send_mail']:
    server.quit()
