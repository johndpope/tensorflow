#export C_INCLUDE_PATH=/usr/local/include
CFLAGS := -I/usr/local/include
LDFLAGS := -L/usr/local/lib 

# LDFLAGS := -L../../bazel-bin/tensorflow
#LDFLAGS := -L../../bazel-out/local-fastbuild/bin/tensorflow
#generate-xcodeproj --xcconfig-overrides settings.xcconfig

SFLAGS=-Xcc $(CFLAGS) -Xlinker $(LDFLAGS)

DBGLIB=.build/debug/TensorFlow
RELLIB=.build/release/TensorFlow.swiftmodule

all: $(DBGLIB)

$(DBGLIB): Sources/*
#	swift package reset
#	swift package clean
	swift build $(SFLAGS)  


release: $(RELLIB)

$(RELLIB): Sources/*
	swift build -c release $(SFLAGS) 

test: $(DBGLIB)
	swift test $(SFLAGS)

clean:
	rm -rf $(DBGLIB) $(RELLIB)
	swift build --clean
