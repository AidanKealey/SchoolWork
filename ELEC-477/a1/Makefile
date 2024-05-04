UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
CXX = g++
CXXFLAGS = -std=c++2a -g -pthread -g
LDFLAGS = -pthread -lprotobuf -lgdbm
else
CXX = g++
CXXFLAGS = -std=c++20 -pthread -g
LDFLAGS = -pthread -lprotobuf
endif

all: bin/a1

# rpc.pb.cc rpc.pb.h: rpc.proto
# 	protoc --cpp_out=. rpc.proto

OBJS=dumpHex.o main.o network.o server.o service.o client.o clientStub.o rpc.pb.o
bin/a1:  $(OBJS)
	$(CXX) -g -o bin/a1 $(OBJS) $(LDFLAGS)

# Add a compilation rule for rpc.pb.o
rpc.pb.o: rpc.pb.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<
	

main.o:  network.hpp dumpHex.hpp server.hpp service.hpp client.hpp clientStub.hpp rpc.pb.h
network.o: network.hpp
dumpHex.o: dumpHex.hpp
server.o: server.hpp service.hpp network.hpp rpc.pb.h
service.o: service.hpp network.hpp rpc.pb.h
client.o: client.hpp clientStub.hpp network.hpp rpc.pb.h
clientStub.o: clientStub.hpp network.hpp rpc.pb.h

clean:
	rm bin/* *.o
