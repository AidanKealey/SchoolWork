#include "network.hpp"
#include "server.hpp"
#include "service.hpp"
#include "client.hpp"
#include "clientStub.hpp"

#include <iostream>
#include <cstdint>
#include "rpc.pb.h"
#include <sstream>
#include <chrono>
#include <thread>
#include <string>
#include <mutex>

extern std::map<std::thread::id, std::shared_ptr<Node>> nodes;
extern std::map<std::thread::id, std::string> names;
extern std::mutex nodes_mutex; 

int main(int argc, char* argv[]) {
    // Handle command line arguments...
    int res = network_init(argc, argv);
    std::stringstream ss;

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: starting server" << std::endl;

    // Start the server
    auto server = std::make_shared<RPCServer>("server");
    dynamic_cast<RPCServer*>(server.get())->setDatabaseName("server_db.gdbm");
    server->setAddress("10.0.0.2"); // Set server address to 10.0.0.2
    server->startServices();

    // Wait for servers to get up and running...
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: initializing client" << std::endl;

    auto client = std::make_shared<RPCClient>("client");
    dynamic_cast<RPCClient*>(client.get())->setServerAddress("10.0.0.3"); 

    std::shared_ptr<std::thread> clientThread;
    {
        std::lock_guard<std::mutex> guard(nodes_mutex);
        clientThread = std::make_shared<std::thread>([client]() {
            try {
                client->start();
            } catch (const std::exception& e) {
                std::cerr << "Client encountered an exception: " << e.what() << std::endl;
            }
        });
        nodes.insert(std::make_pair(clientThread->get_id(), client));
        names.insert(std::make_pair(clientThread->get_id(), "client"));
    }

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: waiting for client to finish" << std::endl;
    clientThread->join();

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: calling stop services on server" << std::endl;
    server->stopServices();

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: waiting for server threads to complete" << std::endl;
    server->waitForServices();

    std::cout << "Main: Shutdown complete." << std::endl;
    google::protobuf::ShutdownProtobufLibrary(); 

    return 0;
}
