========================================================================
    CONSOLE APPLICATION : video-tagging Project Overview
========================================================================

1. Have to copy all file from resource and replace in the same directory of executable file
2. Pre-trained files is in "trainer"

Usage :
For sighthound method
	video-tagging.exe [video file path] sighthound
	Example --> video-tagging.exe C:\mar.mp4 sighthound
For svm method
	video-tagging.exe [video file path] svm [pre-trined directory path]
	Example --> video-tagging.exe C:\mar.mp4 svm C:\trainer 

/////////////////////////////////////////////////////////////////////////////
AppWizard has created this video-tagging application for you.

This file contains a summary of what you will find in each of the files that
make up your video-tagging application.


video-tagging.vcxproj
    This is the main project file for VC++ projects generated using an Application Wizard.
    It contains information about the version of Visual C++ that generated the file, and
    information about the platforms, configurations, and project features selected with the
    Application Wizard.

video-tagging.vcxproj.filters
    This is the filters file for VC++ projects generated using an Application Wizard. 
    It contains information about the association between the files in your project 
    and the filters. This association is used in the IDE to show grouping of files with
    similar extensions under a specific node (for e.g. ".cpp" files are associated with the
    "Source Files" filter).

video-tagging.cpp
    This is the main application source file.

/////////////////////////////////////////////////////////////////////////////
Other standard files:

StdAfx.h, StdAfx.cpp
    These files are used to build a precompiled header (PCH) file
    named video-tagging.pch and a precompiled types file named StdAfx.obj.

/////////////////////////////////////////////////////////////////////////////
Other notes:

AppWizard uses "TODO:" comments to indicate parts of the source code you
should add to or customize.

/////////////////////////////////////////////////////////////////////////////
