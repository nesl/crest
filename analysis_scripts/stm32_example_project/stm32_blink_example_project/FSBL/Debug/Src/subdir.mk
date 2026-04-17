################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Src/main.c \
../Src/stm32n6xx_hal.c \
../Src/stm32n6xx_hal_cortex.c \
../Src/stm32n6xx_hal_dma.c \
../Src/stm32n6xx_hal_dma_ex.c \
../Src/stm32n6xx_hal_exti.c \
../Src/stm32n6xx_hal_gpio.c \
../Src/stm32n6xx_hal_msp.c \
../Src/stm32n6xx_hal_pwr.c \
../Src/stm32n6xx_hal_pwr_ex.c \
../Src/stm32n6xx_hal_rcc.c \
../Src/stm32n6xx_hal_rcc_ex.c \
../Src/stm32n6xx_it.c \
../Src/stm32n6xx_nucleo.c \
../Src/syscalls.c \
../Src/sysmem.c \
../Src/system_stm32n6xx_fsbl.c 

OBJS += \
./Src/main.o \
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
./Src/stm32n6xx_it.o \
./Src/stm32n6xx_nucleo.o \
./Src/syscalls.o \
./Src/sysmem.o \
./Src/system_stm32n6xx_fsbl.o 

C_DEPS += \
./Src/main.d \
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
./Src/stm32n6xx_it.d \
./Src/stm32n6xx_nucleo.d \
./Src/syscalls.d \
./Src/sysmem.d \
./Src/system_stm32n6xx_fsbl.d 

LOCAL_C_INCLUDES := \
-I../Inc \
-I../Drivers/BSP/Components/mx25um51245g \
-I../Drivers/BSP/STM32N6xx_Nucleo \
-I../Drivers/STM32N6xx_HAL_Driver/Inc \
-I../Drivers/CMSIS/Device/ST/STM32N6xx/Include \
-I../Drivers/STM32N6xx_HAL_Driver/Inc/Legacy \
-I../Drivers/CMSIS/Include

# Each subdirectory must supply rules for building sources it contributes
Src/%.o Src/%.su Src/%.cyclo: ../Src/%.c Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32N657xx -c $(LOCAL_C_INCLUDES) -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Src

clean-Src:
	-$(RM) ./Src/main.cyclo ./Src/main.d ./Src/main.o ./Src/main.su ./Src/stm32n6xx_hal.cyclo ./Src/stm32n6xx_hal.d ./Src/stm32n6xx_hal.o ./Src/stm32n6xx_hal.su ./Src/stm32n6xx_hal_cortex.cyclo ./Src/stm32n6xx_hal_cortex.d ./Src/stm32n6xx_hal_cortex.o ./Src/stm32n6xx_hal_cortex.su ./Src/stm32n6xx_hal_dma.cyclo ./Src/stm32n6xx_hal_dma.d ./Src/stm32n6xx_hal_dma.o ./Src/stm32n6xx_hal_dma.su ./Src/stm32n6xx_hal_dma_ex.cyclo ./Src/stm32n6xx_hal_dma_ex.d ./Src/stm32n6xx_hal_dma_ex.o ./Src/stm32n6xx_hal_dma_ex.su ./Src/stm32n6xx_hal_exti.cyclo ./Src/stm32n6xx_hal_exti.d ./Src/stm32n6xx_hal_exti.o ./Src/stm32n6xx_hal_exti.su ./Src/stm32n6xx_hal_gpio.cyclo ./Src/stm32n6xx_hal_gpio.d ./Src/stm32n6xx_hal_gpio.o ./Src/stm32n6xx_hal_gpio.su ./Src/stm32n6xx_hal_msp.cyclo ./Src/stm32n6xx_hal_msp.d ./Src/stm32n6xx_hal_msp.o ./Src/stm32n6xx_hal_msp.su ./Src/stm32n6xx_hal_pwr.cyclo ./Src/stm32n6xx_hal_pwr.d ./Src/stm32n6xx_hal_pwr.o ./Src/stm32n6xx_hal_pwr.su ./Src/stm32n6xx_hal_pwr_ex.cyclo ./Src/stm32n6xx_hal_pwr_ex.d ./Src/stm32n6xx_hal_pwr_ex.o ./Src/stm32n6xx_hal_pwr_ex.su ./Src/stm32n6xx_hal_rcc.cyclo ./Src/stm32n6xx_hal_rcc.d ./Src/stm32n6xx_hal_rcc.o ./Src/stm32n6xx_hal_rcc.su ./Src/stm32n6xx_hal_rcc_ex.cyclo ./Src/stm32n6xx_hal_rcc_ex.d ./Src/stm32n6xx_hal_rcc_ex.o ./Src/stm32n6xx_hal_rcc_ex.su ./Src/stm32n6xx_it.cyclo ./Src/stm32n6xx_it.d ./Src/stm32n6xx_it.o ./Src/stm32n6xx_it.su ./Src/stm32n6xx_nucleo.cyclo ./Src/stm32n6xx_nucleo.d ./Src/stm32n6xx_nucleo.o ./Src/stm32n6xx_nucleo.su ./Src/syscalls.cyclo ./Src/syscalls.d ./Src/syscalls.o ./Src/syscalls.su ./Src/sysmem.cyclo ./Src/sysmem.d ./Src/sysmem.o ./Src/sysmem.su ./Src/system_stm32n6xx_fsbl.cyclo ./Src/system_stm32n6xx_fsbl.d ./Src/system_stm32n6xx_fsbl.o ./Src/system_stm32n6xx_fsbl.su

.PHONY: clean-Src
