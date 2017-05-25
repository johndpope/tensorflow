# see https://www.tensorflow.org/versions/master/install/install_c

CFLAGS := "-I/usr/local/include"
LDFLAGS := "-L/usr/local/lib -ltensorflow" 
SFLAGS=-Xcc $(CFLAGS) -Xlinker $(LDFLAGS)

DBGLIB=.build/debug/TensorFlow1
RELLIB=.build/release/TensorFlow1

all: $(DBGLIB)

$(DBGLIB): Sources/*
#	swift package reset
#	swift package clean
#	swift build $(SFLAGS)  
	swift package generate-xcodeproj --xcconfig-overrides settings.xcconfig


release: $(RELLIB)

$(RELLIB): Sources/*
	swift build -c release $(SFLAGS) 

test: $(DBGLIB)
	swift test $(SFLAGS)

clean:
	rm -rf $(DBGLIB) $(RELLIB)
	swift build --clean
