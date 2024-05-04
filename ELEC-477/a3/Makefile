UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
CXX=g++
CXXFLAGS=-std=c++2a -g -pthread -g
LDFLAGS= -lprotobuf -pthread -lgdbm
endif
ifeq ($(UNAME_S),Darwin)
CXXFLAGS=-std=c++20 -g -pthread -g
CXX=c++
LDFLAGS= -lprotobuf -pthread
endif

# Added kvserverstub.o to the list of object files
OBJS=main.o network.o E477KV.pb.o dumpHex.o kvserver.o kvservice.o kvclient1.o kvclientstub.o E477DirectoryService.pb.o sdservice.o sdserver.o sdstub.o kvserverstub.o

all: bin/assign3
bin/assign3:  $(OBJS)
	$(CXX) -g -o bin/assign3 $(OBJS)  $(LDFLAGS)

main.o:  E477KV.pb.h E477DirectoryService.pb.h

E477KV.pb.h E477KV.pb.cc: E477KV.proto
	protoc --cpp_out=. E477KV.proto

E477DirectoryService.pb.h E477DirectoryService.pb.cc: E477DirectoryService.proto
	protoc --cpp_out=. E477DirectoryService.proto

kvclient1.o: E477KV.pb.h kvclient1.hpp KVrpc.h kvclientstub.hpp network.hpp
kvclientstub.o: kvclientstub.hpp network.hpp

kvservice.o: kvservice.hpp E477KV.pb.h network.hpp #E477.h
kvserver.o: kvserver.hpp network.hpp kvservice.hpp

# Added dependency for kvserverstub.o
kvserverstub.o: kvserverstub.hpp E477KV.pb.h network.hpp

sdservice.o: sdservice.hpp network.hpp E477DirectoryService.pb.h
sdserver.o: sdserver.hpp network.hpp sdservice.hpp
sdstub.o: sdstub.hpp network.hpp E477DirectoryService.pb.h

# support files
network.o: network.hpp dumpHex.hpp
dumpHex.o: dumpHex.hpp

clean:
	rm -f bin/* *.o

# Uncommented and updated clean command to also remove generated protobuf files if needed
# clean:
#   rm -f bin/* *.o E477KV.pb.h E477KV.pb.cc E477DirectoryService.pb.h E477DirectoryService.pb.cc
