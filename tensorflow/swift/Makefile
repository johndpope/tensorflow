# see https://www.tensorflow.org/versions/master/install/install_c

#  --xcconfig-overrides settings.xcconfig  - targets only x86
LDFLAGS := -L/usr/local/lib # -llibtensorflow.so
CFLAGS := -I/usr/local/include

SFLAGS=-Xcc $(CFLAGS) -Xlinker $(LDFLAGS)

DBGLIB=.build/debug/TensorFlow
RELLIB=.build/release/TensorFlow

all: $(DBGLIB)

$(DBGLIB): Sources/*
	rm -rf .build
	rm Package.pins
	swift package reset
	swift package clean
	swift package $(SFLAGS)  generate-xcodeproj --xcconfig-overrides settings.xcconfig
	
release: $(RELLIB)

$(RELLIB): Sources/*
	swift build -c release $(SFLAGS)

test: $(DBGLIB)
	swift test $(SFLAGS)

clean:
	rm -rf $(DBGLIB) $(RELLIB)
	swift package clean
