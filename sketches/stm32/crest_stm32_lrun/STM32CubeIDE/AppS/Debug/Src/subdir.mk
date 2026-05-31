################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

-include ../stedgeai.mk

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../../../Appli/Src/main.c \
../../../Appli/Src/mx25um51245g.c \
../../../Appli/Src/network.c \
../../../Appli/Src/network_data.c \
../../../Appli/Src/network_data_params.c \
../../../Appli/Src/secure_nsc.c \
../../../Appli/Src/stm32n6xx_hal.c \
../../../Appli/Src/stm32n6xx_hal_cortex.c \
../../../Appli/Src/stm32n6xx_hal_dma.c \
../../../Appli/Src/stm32n6xx_hal_dma_ex.c \
../../../Appli/Src/stm32n6xx_hal_exti.c \
../../../Appli/Src/stm32n6xx_hal_gpio.c \
../../../Appli/Src/stm32n6xx_hal_msp.c \
../../../Appli/Src/stm32n6xx_hal_pwr.c \
../../../Appli/Src/stm32n6xx_hal_pwr_ex.c \
../../../Appli/Src/stm32n6xx_hal_rcc.c \
../../../Appli/Src/stm32n6xx_hal_rcc_ex.c \
../../../Appli/Src/stm32n6xx_hal_rtc.c \
../../../Appli/Src/stm32n6xx_hal_rtc_ex.c \
../../../Appli/Src/stm32n6xx_hal_uart.c \
../../../Appli/Src/stm32n6xx_hal_uart_ex.c \
../../../Appli/Src/stm32n6xx_hal_xspi.c \
../../../Appli/Src/stm32n6xx_it.c \
../../../Appli/Src/stm32n6xx_nucleo.c \
../../../Appli/Src/stm32n6xx_nucleo_xspi.c \
../Src/syscalls.c \
../Src/sysmem.c \
../../../Appli/Src/system_stm32n6xx_s.c \
../../../Appli/Src/crest_dut_runner.c

OBJS += \
./Src/main.o \
./Src/mx25um51245g.o \
./Src/network.o \
./Src/network_data.o \
./Src/network_data_params.o \
./Src/secure_nsc.o \
./Src/stm32n6xx_hal.o \
./Src/stm32n6xx_hal_cortex.o \
./Src/stm32n6xx_hal_dma.o \
./Src/stm32n6xx_hal_dma_ex.o \
./Src/stm32n6xx_hal_exti.o \
./Src/stm32n6xx_hal_gpio.o \
./Src/stm32n6xx_hal_msp.o \
./Src/stm32n6xx_hal_pwr.o \
./Src/stm32n6xx_hal_pwr_ex.o \
./Src/stm32n6xx_hal_rcc.o \
./Src/stm32n6xx_hal_rcc_ex.o \
./Src/stm32n6xx_hal_rtc.o \
./Src/stm32n6xx_hal_rtc_ex.o \
./Src/stm32n6xx_hal_uart.o \
./Src/stm32n6xx_hal_uart_ex.o \
./Src/stm32n6xx_hal_xspi.o \
./Src/stm32n6xx_it.o \
./Src/stm32n6xx_nucleo.o \
./Src/stm32n6xx_nucleo_xspi.o \
./Src/syscalls.o \
./Src/sysmem.o \
./Src/system_stm32n6xx_s.o \
./Src/crest_dut_runner.o

C_DEPS += \
./Src/main.d \
./Src/mx25um51245g.d \
./Src/network.d \
./Src/network_data.d \
./Src/network_data_params.d \
./Src/secure_nsc.d \
./Src/stm32n6xx_hal.d \
./Src/stm32n6xx_hal_cortex.d \
./Src/stm32n6xx_hal_dma.d \
./Src/stm32n6xx_hal_dma_ex.d \
./Src/stm32n6xx_hal_exti.d \
./Src/stm32n6xx_hal_gpio.d \
./Src/stm32n6xx_hal_msp.d \
./Src/stm32n6xx_hal_pwr.d \
./Src/stm32n6xx_hal_pwr_ex.d \
./Src/stm32n6xx_hal_rcc.d \
./Src/stm32n6xx_hal_rcc_ex.d \
./Src/stm32n6xx_hal_rtc.d \
./Src/stm32n6xx_hal_rtc_ex.d \
./Src/stm32n6xx_hal_uart.d \
./Src/stm32n6xx_hal_uart_ex.d \
./Src/stm32n6xx_hal_xspi.d \
./Src/stm32n6xx_it.d \
./Src/stm32n6xx_nucleo.d \
./Src/stm32n6xx_nucleo_xspi.d \
./Src/syscalls.d \
./Src/sysmem.d \
./Src/system_stm32n6xx_s.d \
./Src/crest_dut_runner.d

LOCAL_C_INCLUDES = \
-I../../../Appli/Inc \
-I../../../Secure_nsclib \
-I../../../Drivers/BSP/Components/mx25um51245g \
-I../../../Drivers/BSP/STM32N6xx_Nucleo \
-I../../../Drivers/STM32N6xx_HAL_Driver/Inc \
-I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include \
-I../../../Drivers/STM32N6xx_HAL_Driver/Inc/Legacy \
-I../../../Drivers/CMSIS/Include \
-I$(STEDGEAI_INC)

# Each subdirectory must supply rules for building sources it contributes
Src/%.o Src/%.su Src/%.cyclo: ../../../Appli/Src/%.c Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32N657xx -c $(LOCAL_C_INCLUDES) -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

Src/mx25um51245g.o Src/mx25um51245g.su Src/mx25um51245g.cyclo: ../../../Appli/Src/mx25um51245g.c Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32N657xx -c $(LOCAL_C_INCLUDES) -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

Src/stm32n6xx_hal_xspi.o Src/stm32n6xx_hal_xspi.su Src/stm32n6xx_hal_xspi.cyclo: ../../../Appli/Src/stm32n6xx_hal_xspi.c Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32N657xx -c $(LOCAL_C_INCLUDES) -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

Src/stm32n6xx_hal_rtc.o Src/stm32n6xx_hal_rtc.su Src/stm32n6xx_hal_rtc.cyclo: ../../../Appli/Src/stm32n6xx_hal_rtc.c Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32N657xx -c $(LOCAL_C_INCLUDES) -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

Src/stm32n6xx_hal_rtc_ex.o Src/stm32n6xx_hal_rtc_ex.su Src/stm32n6xx_hal_rtc_ex.cyclo: ../../../Appli/Src/stm32n6xx_hal_rtc_ex.c Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32N657xx -c $(LOCAL_C_INCLUDES) -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

Src/stm32n6xx_nucleo_xspi.o Src/stm32n6xx_nucleo_xspi.su Src/stm32n6xx_nucleo_xspi.cyclo: ../../../Appli/Src/stm32n6xx_nucleo_xspi.c Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32N657xx -c $(LOCAL_C_INCLUDES) -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

Src/syscalls.o Src/syscalls.su Src/syscalls.cyclo: ../Src/syscalls.c Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32N657xx -c $(LOCAL_C_INCLUDES) -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

Src/sysmem.o Src/sysmem.su Src/sysmem.cyclo: ../Src/sysmem.c Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32N657xx -c $(LOCAL_C_INCLUDES) -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Src

clean-Src:
	-$(RM) ./Src/main.cyclo ./Src/main.d ./Src/main.o ./Src/main.su ./Src/mx25um51245g.cyclo ./Src/mx25um51245g.d ./Src/mx25um51245g.o ./Src/mx25um51245g.su ./Src/network.cyclo ./Src/network.d ./Src/network.o ./Src/network.su ./Src/network_data.cyclo ./Src/network_data.d ./Src/network_data.o ./Src/network_data.su ./Src/network_data_params.cyclo ./Src/network_data_params.d ./Src/network_data_params.o ./Src/network_data_params.su ./Src/secure_nsc.cyclo ./Src/secure_nsc.d ./Src/secure_nsc.o ./Src/secure_nsc.su ./Src/stm32n6xx_hal.cyclo ./Src/stm32n6xx_hal.d ./Src/stm32n6xx_hal.o ./Src/stm32n6xx_hal.su ./Src/stm32n6xx_hal_cortex.cyclo ./Src/stm32n6xx_hal_cortex.d ./Src/stm32n6xx_hal_cortex.o ./Src/stm32n6xx_hal_cortex.su ./Src/stm32n6xx_hal_dma.cyclo ./Src/stm32n6xx_hal_dma.d ./Src/stm32n6xx_hal_dma.o ./Src/stm32n6xx_hal_dma.su ./Src/stm32n6xx_hal_dma_ex.cyclo ./Src/stm32n6xx_hal_dma_ex.d ./Src/stm32n6xx_hal_dma_ex.o ./Src/stm32n6xx_hal_dma_ex.su ./Src/stm32n6xx_hal_exti.cyclo ./Src/stm32n6xx_hal_exti.d ./Src/stm32n6xx_hal_exti.o ./Src/stm32n6xx_hal_exti.su ./Src/stm32n6xx_hal_gpio.cyclo ./Src/stm32n6xx_hal_gpio.d ./Src/stm32n6xx_hal_gpio.o ./Src/stm32n6xx_hal_gpio.su ./Src/stm32n6xx_hal_msp.cyclo ./Src/stm32n6xx_hal_msp.d ./Src/stm32n6xx_hal_msp.o ./Src/stm32n6xx_hal_msp.su ./Src/stm32n6xx_hal_pwr.cyclo ./Src/stm32n6xx_hal_pwr.d ./Src/stm32n6xx_hal_pwr.o ./Src/stm32n6xx_hal_pwr.su ./Src/stm32n6xx_hal_pwr_ex.cyclo ./Src/stm32n6xx_hal_pwr_ex.d ./Src/stm32n6xx_hal_pwr_ex.o ./Src/stm32n6xx_hal_pwr_ex.su ./Src/stm32n6xx_hal_rcc.cyclo ./Src/stm32n6xx_hal_rcc.d ./Src/stm32n6xx_hal_rcc.o ./Src/stm32n6xx_hal_rcc.su ./Src/stm32n6xx_hal_rcc_ex.cyclo ./Src/stm32n6xx_hal_rcc_ex.d ./Src/stm32n6xx_hal_rcc_ex.o ./Src/stm32n6xx_hal_rcc_ex.su ./Src/stm32n6xx_hal_rtc.cyclo ./Src/stm32n6xx_hal_rtc.d ./Src/stm32n6xx_hal_rtc.o ./Src/stm32n6xx_hal_rtc.su ./Src/stm32n6xx_hal_rtc_ex.cyclo ./Src/stm32n6xx_hal_rtc_ex.d ./Src/stm32n6xx_hal_rtc_ex.o ./Src/stm32n6xx_hal_rtc_ex.su ./Src/stm32n6xx_hal_uart.cyclo ./Src/stm32n6xx_hal_uart.d ./Src/stm32n6xx_hal_uart.o ./Src/stm32n6xx_hal_uart.su ./Src/stm32n6xx_hal_uart_ex.cyclo ./Src/stm32n6xx_hal_uart_ex.d ./Src/stm32n6xx_hal_uart_ex.o ./Src/stm32n6xx_hal_uart_ex.su ./Src/stm32n6xx_hal_xspi.cyclo ./Src/stm32n6xx_hal_xspi.d ./Src/stm32n6xx_hal_xspi.o ./Src/stm32n6xx_hal_xspi.su ./Src/stm32n6xx_it.cyclo ./Src/stm32n6xx_it.d ./Src/stm32n6xx_it.o ./Src/stm32n6xx_it.su ./Src/stm32n6xx_nucleo.cyclo ./Src/stm32n6xx_nucleo.d ./Src/stm32n6xx_nucleo.o ./Src/stm32n6xx_nucleo.su ./Src/stm32n6xx_nucleo_xspi.cyclo ./Src/stm32n6xx_nucleo_xspi.d ./Src/stm32n6xx_nucleo_xspi.o ./Src/stm32n6xx_nucleo_xspi.su ./Src/syscalls.cyclo ./Src/syscalls.d ./Src/syscalls.o ./Src/syscalls.su ./Src/sysmem.cyclo ./Src/sysmem.d ./Src/sysmem.o ./Src/sysmem.su ./Src/system_stm32n6xx_s.cyclo ./Src/system_stm32n6xx_s.d ./Src/system_stm32n6xx_s.o ./Src/system_stm32n6xx_s.su ./Src/crest_dut_runner.cyclo ./Src/crest_dut_runner.d ./Src/crest_dut_runner.o ./Src/crest_dut_runner.su

.PHONY: clean-Src
