# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Execute the real harness state machine with deterministic GPIO/power stubs."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_harness_counts_only_completed_pulses(tmp_path):
    """Exercise pulse accumulation and legacy completion on the host.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated compiler workspace.
    """
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('g++ is required for the harness state-machine test')
    firmware = Path(__file__).resolve().parents[1]/'firmware'
    (tmp_path/'common').mkdir()
    shutil.copy2(firmware/'harness/harness.ino', tmp_path/'harness.cc')
    shutil.copy2(firmware/'common/crest_hil_config.h', tmp_path/'common/crest_hil_config.h')
    (tmp_path/'common/crest_power.h').write_text(r'''
#include <cstdint>
#include <cmath>
#include <string>
#include <sstream>
using std::isfinite;
constexpr int LOW=0, HIGH=1, INPUT_PULLUP=2, INPUT_PULLDOWN=3;
unsigned now_us=1000; int arm=HIGH, trigger=LOW;
unsigned millis(){return now_us/1000;} unsigned micros(){return now_us;}
void delay(unsigned ms){now_us+=ms*1000;} void pinMode(int,int){}
int digitalRead(int pin){return pin==3?arm:trigger;}
struct String:std::string {
 using std::string::string;
 void trim(){} bool equalsIgnoreCase(const char* s) const {return *this==s;}
};
struct SerialStub {
 std::ostringstream log;
 void begin(int){} void setTimeout(unsigned){} operator bool(){return true;}
 int available(){return 0;} String readStringUntil(char){return String();}
 template<class T> void print(T v){log<<v;}
 template<class T> void println(T v){log<<v<<'\n';}
 void println(float v,int){log<<v<<'\n';}
} Serial;
namespace crest_power {
int begins=0, ends=0, flushes=0; float emitted_energy=0; unsigned emitted_runs=0;
void ResetAccumulators(){} void BeginInvocationWindow(){++begins;}
float EndInvocationWindow(){++ends;return 2.0f;}
float FlushEnergyWindow(){++flushes;return 5.0f;}
void ConfigureInvocationWindows(bool){} void InitializePowerMonitor(){}
void EmitInvocationWindowSettings(){}
float GetIdleBaselinePower(){return 0;}
void EmitPowerTelemetry(unsigned,float,int runs,float,float energy){emitted_runs=runs;emitted_energy=energy;}
}
''')
    (tmp_path/'main.cc').write_text(r'''
#include <cassert>
#include "harness.cc"
void step(unsigned us,int a,int t){now_us+=us;arm=a;trigger=t;loop();}
int main(){
 setup(); invocation_windows=true;
 step(1000,LOW,LOW);step(600000,LOW,LOW);
 assert(state==HarnessState::kArmed);
 step(200000,LOW,LOW); // warmup: no energy window
 assert(crest_power::begins==0);
 step(1,LOW,HIGH);step(200000,LOW,LOW);
 step(20000,LOW,LOW); // UART/yield: excluded
 step(1,LOW,HIGH);step(300000,LOW,LOW);
 step(20000,HIGH,LOW);
 assert(crest_power::begins==2 && crest_power::ends==2);
 assert(crest_power::emitted_runs==2 && crest_power::emitted_energy==4.0f);
 assert(window_us==500000 && window_count==2);
 assert(Serial.log.str().find("energy windows: 2 elapsed_us: 500000")!=std::string::npos);
 // Default protocol still finishes on its first falling edge.
 invocation_windows=false;EnterIdle();
 step(1000,LOW,LOW);step(600000,LOW,LOW);
 step(1,LOW,HIGH);step(200000,LOW,LOW);
 assert(crest_power::flushes==1);
 assert(crest_power::emitted_runs==CREST_INFERENCE_RUNS);
}
''')
    subprocess.run([compiler, '-std=c++17', '-I', str(tmp_path), str(tmp_path/'main.cc'),
                    '-o', str(tmp_path/'test')], check=True, capture_output=True)
    subprocess.run([str(tmp_path/'test')], check=True, capture_output=True)
