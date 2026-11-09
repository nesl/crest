################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal.c \
../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_bsec.c \
../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_cortex.c \
../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_dma.c \
../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_gpio.c \
../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_pwr.c \
../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_pwr_ex.c \
../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rcc.c \
../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rcc_ex.c \
../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_xspi.c 

OBJS += \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal.o \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_bsec.o \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_cortex.o \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_dma.o \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_gpio.o \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr.o \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr_ex.o \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc.o \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc_ex.o \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_xspi.o 

C_DEPS += \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal.d \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_bsec.d \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_cortex.d \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_dma.d \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_gpio.d \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr.d \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr_ex.d \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc.d \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc_ex.d \
./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_xspi.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal.o: ../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal.c Drivers/STM32N6xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_bsec.o: ../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_bsec.c Drivers/STM32N6xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_cortex.o: ../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_cortex.c Drivers/STM32N6xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_dma.o: ../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_dma.c Drivers/STM32N6xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_gpio.o: ../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_gpio.c Drivers/STM32N6xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr.o: ../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_pwr.c Drivers/STM32N6xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr_ex.o: ../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_pwr_ex.c Drivers/STM32N6xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc.o: ../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rcc.c Drivers/STM32N6xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc_ex.o: ../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rcc_ex.c Drivers/STM32N6xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_xspi.o: ../../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_xspi.c Drivers/STM32N6xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-STM32N6xx_HAL_Driver

clean-Drivers-2f-STM32N6xx_HAL_Driver:
	-$(RM) ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal.cyclo ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal.d ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal.o ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal.su ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_bsec.cyclo ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_bsec.d ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_bsec.o ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_bsec.su ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_cortex.cyclo ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_cortex.d ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_cortex.o ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_cortex.su ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_dma.cyclo ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_dma.d ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_dma.o ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_dma.su ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_gpio.cyclo ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_gpio.d ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_gpio.o ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_gpio.su ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr.cyclo ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr.d ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr.o ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr.su ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr_ex.cyclo ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr_ex.d ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr_ex.o ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_pwr_ex.su ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc.cyclo ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc.d ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc.o ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc.su ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc_ex.cyclo ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc_ex.d ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc_ex.o ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_rcc_ex.su ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_xspi.cyclo ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_xspi.d ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_xspi.o ./Drivers/STM32N6xx_HAL_Driver/stm32n6xx_hal_xspi.su

.PHONY: clean-Drivers-2f-STM32N6xx_HAL_Driver

