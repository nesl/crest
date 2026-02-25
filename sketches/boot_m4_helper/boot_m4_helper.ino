#include <Arduino.h>
#include <RPC.h>

void setup() {
  RPC.begin();  // Boots M4 and keeps inter-core plumbing initialized.
}

void loop() {
  // Light CPU idle only: wake on interrupt (USB/RPC/SysTick), avoid deep sleep.
  __WFI();
}
