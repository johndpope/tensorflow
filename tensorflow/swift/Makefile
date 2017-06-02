# see https://www.tensorflow.org/versions/master/install/install_c

CFLAGS = -Xcc -I/usr/local/include
LDFLAGS = -Xlinker "-L/usr/local/lib -ltensorflow"

DBGLIB=.build/debug/TensorFlow1
RELLIB=.build/release/TensorFlow1

all: $(DBGLIB)

$(DBGLIB): Sources/*
#	swift package reset
#	swift package clean
#	swift build -v $(CFLAGS) $(LDFLAGS) 
	swift package generate-xcodeproj --xcconfig-overrides settings.xcconfig


test:
	swift build -v $(CFLAGS) $(LDFLAGS)
	swift test -v $(CFLAGS) $(LDFLAGS) 

clean:
	swift package clean
	rm -rf Packages
	rm -rf .build
	rm -rf Package.pins