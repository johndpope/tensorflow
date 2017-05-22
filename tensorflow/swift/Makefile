# see https://www.tensorflow.org/versions/master/install/install_c



LDFLAGS := -L/usr/local/lib 
CFLAGS := -I/usr/local/include

SFLAGS=-Xcc $(CFLAGS) -Xlinker $(LDFLAGS)

DBGLIB=.build/debug/TensorFlow.swiftmodule
RELLIB=.build/release/TensorFlow.swiftmodule

all: $(DBGLIB)

$(DBGLIB): Sources/*
#	echo "import CTensorFlow\n:print_decl $2" | xcrun swift -deprecated-integrated-repl $(SFLAGS) - doesn't work
# https://pewpewthespells.com/blog/custom_frameworks_and_swift.html
# var handle = dlopen(".build/debug/TensorFlow.swiftmodule", 2)

# Under the heading "Apple LLVM 6.0 - Language - Modules" enable the setting "Enable Modules (C and Objective-C)". This will create the module file needed for swift to auto-generate the swift headers from C and Objective-C headers.
# OR 
# --xcconfig-overrides  TensorFlow.xcconfig  
	echo -e "import CTensorFlow\n:type lookup CTensorFlow" | swift  $(SFLAGS)  
	swift package -Xswiftc -framework -Xswiftc "DEBUG" $(SFLAGS) generate-xcodeproj --xcconfig-overrides settings.xcconfig
# 	xcodebuild -scheme "TensorFlow" -showBuildSettings >> TensorFlow.xcconfig
# 
#	swift build $(SFLAGS)
#	echo ".build/debug/TensorFlow.swiftmodule"
	
release: $(RELLIB)

$(RELLIB): Sources/*
	swift build -c release $(SFLAGS)

test: $(DBGLIB)
	swift test $(SFLAGS)

clean:
	rm -rf $(DBGLIB) $(RELLIB)
	swift build --clean
