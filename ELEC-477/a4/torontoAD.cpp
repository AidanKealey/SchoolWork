#include <cstdlib>
#include <iostream>
#include <chrono>
#include <thread>
#include <set>
#include <cmath>

#include "dds/dds.hpp"

#include "statekey.hpp"
#include "zoneTransfer.hpp"


using namespace std;
using namespace org::eclipse::cyclonedds;

// toronto airport coords: 43.6771° N, 79.6334° W
const double lat_air_D = 43.6771;
const double lon_air_D = -79.6334;
const double lat_air_R = 43.6771 * M_PI / 180.0;
const double lon_air_R = -79.6334 * M_PI / 180.0;
const double rad = (360 * M_PI / 180.0);

set<string> planeInBoundaryBuffer;


// set this to your team number
uint32_t domainID = 25;
char * programName;

double getDistance(double lat_plane, double lon_plane) {
    lat_plane = lat_plane * M_PI / 180.0;
    lon_plane = lon_plane * M_PI / 180.0;

    double dlat = lat_plane - lat_air_R;
    double dlon = lon_plane - lon_air_R;
    double a = pow(sin(dlat / 2), 2) + cos(lat_air_R) * cos(lat_plane) * pow(sin(dlon / 2), 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    double distance = (6371 * c) / 1.852;

    return distance;
}

string getDirection(double lat_plane, double lon_plane, double heading) {
    bool direction;
    string directionOfTravel;

    if (lat_plane > lat_air_D) { // plane is NORTH of airport
        if (lon_plane > lon_air_D) { // plane is EAST of the airport
            if (180 >= heading && heading <= 270) { // plane is heading TOWARDS the airport
                direction = true;
            } else { // plane is heading AWAY from the airport
                direction = false;
            }
        } else { // plane must be WEST of the airport
            if (270 >= heading && heading <= 360) { // plane is heading TOWARDS the airport
                direction = true;
            } else { // plane is heading AWAY from the airport
                direction = false;
            }
        }
    } else { // plane must be SOUTH of the airport
        if (lon_plane > lon_air_D) { // plane is EAST of the airport
            if (90 >= heading && heading <= 180) { // plane is heading TOWARDS the airport
                direction = true;
            } else { // plane is heading AWAY from the airport
                direction = false;
            }
        } else { // plane must be WEST of the airport
            if (0 >= heading && heading <= 90) { // plane is heading TOWARDS the airport
                direction = true;
            } else { // plane is heading AWAY from the airport
                direction = false;
            }
        }
    }

    if (direction == true) {
        directionOfTravel = "TOWARDS";
    } else {
        directionOfTravel = "AWAY from";
    }

    return directionOfTravel;
}

int main(int argc, char * argv[]) {
    int count = 0;
    
    programName = argv[0];
    
    // create the main DDS entities Participant, Topic, subTorontoAD and DataReader
    dds::domain::DomainParticipant participant(domainID);
    dds::topic::Topic<State::Update> topic(participant, "Flights");
    // create DDS for TorontoCentre
    dds::topic::Topic<Radar::Route> topic_AD(participant, "TorontoAD");
    dds::topic::Topic<Radar::Route> topic_TC(participant, "TorontoCentre");

    dds::pub::Publisher publisher(participant);
    dds::sub::Subscriber subTorontoAD(participant);

    dds::sub::DataReader<State::Update> reader(subTorontoAD, topic);
    dds::sub::DataReader<Radar::Route> reader2(subTorontoAD, topic_TC);
    dds::pub::DataWriter<Radar::Route> writer(publisher, topic_AD);

    cout << "**** torontoAD waiting for messages" << endl;
    dds::core::cond::WaitSet waitset;
    dds::core::cond::StatusCondition rsc(reader);
    rsc.enabled_statuses(dds::core::status::StatusMask::data_available()| dds::core::status::StatusMask::subscription_matched());
    waitset.attach_condition(rsc);
    
    while(1){
        try{
            // wait for more data or for the publisher to end.
            waitset.wait(dds::core::Duration::infinite());
        } 
        catch (const dds::core::Exception &e){
                cerr << programName << ": subTorontoAD excption while waiting for data: \"" << e.what() << "\"." << endl;
                break;
        }
            
        // take the samples and print them
        dds::sub::LoanedSamples<State::Update> samples;
        samples = reader.take();

        dds::sub::LoanedSamples<Radar::Route> samples2;
        samples2 = reader2.take();

        if (samples.length() > 0){
            dds::sub::LoanedSamples<State::Update>::const_iterator sample_iter;
            for (sample_iter = samples.begin(); sample_iter < samples.end(); ++sample_iter) {
                const State::Update& msg = sample_iter->data();
                const dds::sub::SampleInfo& info = sample_iter->info();
                // note not all samples may be valid.
                if (info.valid()) {
                    if (samples2.length() > 0) {
                        dds::sub::LoanedSamples<Radar::Route>::const_iterator sample2_iter;
                        for (sample2_iter = samples2.begin(); sample2_iter < samples2.end(); ++sample2_iter) {
                            const Radar::Route& alert = sample2_iter->data();
                            const dds::sub::SampleInfo& info2 = sample2_iter->info();
                            // note not all samples may be valid.
                            if (info2.valid()) {
                                if (planeInBoundaryBuffer.find(alert.callsign()) == planeInBoundaryBuffer.end()) {
                                    planeInBoundaryBuffer.insert(alert.callsign());
                                    if (alert.birthplace() == Radar::zone::centre && alert.destination() == Radar::zone::AD) {
                                        cout << "\n**** Received alert from TorontoCentre \n"
                                             << "  |  Prepare for \"" << alert.callsign() << "\" transfer \n"
                                             << "  |  Flight : \"" << alert.callsign() << "\" \n" 
                                             << "  |  Source : \"" << alert.birthplace() << "\" \n"
                                             << "  |  Destination : \"" << alert.destination() << "\" \n"
                                             << "  |  Timestamp : \"" << alert.timestamp() << "\" \n" 
                                             << endl;
                                    }
                                }
                            }
                        }
                    }

                    cout << "**** subTorontoAD received: " << ++count << endl;

                    double distance = getDistance(msg.lat(), msg.lon());
                    double flightHeightft = msg.geoaltitude() * 3.281;
                    string direction = getDirection(msg.lat(), msg.lon(), msg.heading());

                    if (distance > 7.5 && distance <= 8 && flightHeightft <= 3000) {
                        if (planeInBoundaryBuffer.find(msg.callsign()) == planeInBoundaryBuffer.end()) {
                            planeInBoundaryBuffer.insert(msg.callsign());
                            cout << "\n----TorontoAD Boundary Alert---- \n"
                                 << "\nAircraft leaving Toronto AD by crossing the 8nm barrier\n"
                                 << "  |  Flight : \"" << msg.callsign() << "\" \n" 
                                 << "  |  Distance from airport : \"" << distance << "nm\" \n" 
                                 << "  |  The plane is heading " << direction << " the airport \n"
                                 << "  |  Geo Altitude : \"" << flightHeightft << "ft\" \n"
                                 << endl;

                            // send message to Centre
                            Radar::Route alert(msg.callsign(), Radar::zone::AD, Radar::zone::centre, msg.timestamp());
                            writer.write(alert);
                        }
                    } else if (distance <= 8 && (flightHeightft >= 2500 && flightHeightft <= 3000)){
                        if (msg.vertrate() > 0){
                            if (planeInBoundaryBuffer.find(msg.callsign()) == planeInBoundaryBuffer.end()) {
                                planeInBoundaryBuffer.insert(msg.callsign());
                                // plane is exiting AD before 8nm boundary
                                cout << "\n----TorontoAD Boundary Alert---- \n" 
                                     << "\nAircraft leaving Toronto AD by ascending through 3000ft\n"
                                     << "  |  Flight : \"" << msg.callsign() << "\" -- is climbing \n" 
                                     << "  |  Distance from airport : \"" << distance << "nm\" \n"
                                     << "  |  The plane is heading " << direction << " the airport \n" 
                                     << "  |  Geo Altitude : \"" << flightHeightft << "ft\" \n" 
                                     << endl;

                                // send message to Centre 
                                Radar::Route alert(msg.callsign(), Radar::zone::AD, Radar::zone::centre, msg.timestamp());
                                writer.write(alert);
                            }
                        }
                    }
                    // remove from set
                    if (((distance < 7.5 && flightHeightft > 3000) || distance > 8) && (planeInBoundaryBuffer.find(msg.callsign()) != planeInBoundaryBuffer.end())) {
                        planeInBoundaryBuffer.erase(msg.callsign());
                    }
                }
            }
        } else {
            cout << programName << ": no samples?" << endl;
        }

        // if the publisher is gone, exit.
        if (reader.subscription_matched_status().current_count() == 0) {
            break;
        }

        if (reader2.subscription_matched_status().current_count() == 0) {
            break;
        }
    
    }

}
