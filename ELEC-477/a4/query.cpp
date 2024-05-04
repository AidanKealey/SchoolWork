#include <cstdlib>
#include <iostream>
#include <chrono>
#include <thread>
#include <set>
#include <cmath>

#include "dds/dds.hpp"

#include "zoneTransfer.hpp"


using namespace std;
using namespace org::eclipse::cyclonedds;

// set this to your team number
uint32_t domainID = 25;
char * programName;

int main(int argc, char * argv[]) {
    programName = argv[0];

    string callsign;
    cout << "Callsign Options Are: AC180, KM20, PT130, WJ910" << endl;
    cout << "Enter callsign to monitor: ";
    cin >> callsign;
    
    // create the main DDS entities Participant, Topic, subTorontoAD and DataReader
    dds::domain::DomainParticipant participant(domainID);
    // create DDS for TorontoCentre
    dds::topic::Topic<Radar::Route> topic_AD(participant, "TorontoAD");
    dds::topic::Topic<Radar::Route> topic_TC(participant, "TorontoCentre");

    dds::sub::Subscriber sub(participant);

    dds::sub::DataReader<Radar::Route> reader(sub, topic_AD);
    dds::sub::DataReader<Radar::Route> reader2(sub, topic_TC);

    cout << "**** query waiting for messages" << endl;
    dds::core::cond::WaitSet waitset;
    dds::core::cond::StatusCondition rsc(reader);
    dds::core::cond::StatusCondition rsc2(reader2);
    rsc.enabled_statuses(dds::core::status::StatusMask::data_available());
    rsc2.enabled_statuses(dds::core::status::StatusMask::data_available());
    waitset.attach_condition(rsc);
    waitset.attach_condition(rsc2);

    while(1){
        try{
            // wait for more data or for the publisher to end.
            waitset.wait(dds::core::Duration::infinite());
        } 
        catch (const dds::core::Exception &e){
                cerr << programName << ": Subscriber excption while waiting for data: \"" << e.what() << "\"." << endl;
                break;
        }

        dds::sub::LoanedSamples<Radar::Route> samples;
        samples = reader.take();
        if (samples.length() > 0){
            dds::sub::LoanedSamples<Radar::Route>::const_iterator sample_iter;
            for (sample_iter = samples.begin(); sample_iter < samples.end(); ++sample_iter) {
                const Radar::Route& alert = sample_iter->data();
                const dds::sub::SampleInfo& info = sample_iter->info();
                // note not all samples may be valid.
                if (info.valid()) { 
                    if (alert.callsign() == callsign) {
                        cout << "\n**** Received alert from TorontoAD \n"
                             << "  |  Flight : \"" << alert.callsign() << "\" \n" 
                             << "  |  Source : \"" << alert.birthplace() << "\" \n"
                             << "  |  Destination : \"" << alert.destination() << "\" \n"
                             << "  |  Timestamp : \"" << alert.timestamp() << "\" \n" 
                             << endl;
                    }
                }
            }
        }

        samples = reader2.take();
        if (samples.length() > 0){
            dds::sub::LoanedSamples<Radar::Route>::const_iterator sample_iter;
            for (sample_iter = samples.begin(); sample_iter < samples.end(); ++sample_iter) {
                const Radar::Route& alert = sample_iter->data();
                const dds::sub::SampleInfo& info = sample_iter->info();
                // note not all samples may be valid.
                if (info.valid()) { 
                    if (alert.callsign() == callsign) {
                        cout << "\n**** Received alert from TorontoCentre \n"
                             << "  |  Flight : \"" << alert.callsign() << "\" \n" 
                             << "  |  Source : \"" << alert.birthplace() << "\" \n"
                             << "  |  Destination : \"" << alert.destination() << "\" \n"
                             << "  |  Timestamp : \"" << alert.timestamp() << "\" \n" 
                             << endl;
                    }
                }
            }
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