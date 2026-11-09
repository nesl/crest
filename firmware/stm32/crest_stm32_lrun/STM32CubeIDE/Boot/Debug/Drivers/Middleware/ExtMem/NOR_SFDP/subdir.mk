################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../../../Middlewares/ST/STM32_ExtMem_Manager/nor_sfdp/stm32_sfdp_data.c \
../../../Middlewares/ST/STM32_ExtMem_Manager/nor_sfdp/stm32_sfdp_driver.c 

OBJS += \
./Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_data.o \
./Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_driver.o 

C_DEPS += \
./Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_data.d \
./Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_driver.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_data.o: ../../../Middlewares/ST/STM32_ExtMem_Manager/nor_sfdp/stm32_sfdp_data.c Drivers/Middleware/ExtMem/NOR_SFDP/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_driver.o: ../../../Middlewares/ST/STM32_ExtMem_Manager/nor_sfdp/stm32_sfdp_driver.c Drivers/Middleware/ExtMem/NOR_SFDP/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-Middleware-2f-ExtMem-2f-NOR_SFDP

clean-Drivers-2f-Middleware-2f-ExtMem-2f-NOR_SFDP:
	-$(RM) ./Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_data.cyclo ./Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_data.d ./Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_data.o ./Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_data.su ./Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_driver.cyclo ./Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_driver.d ./Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_driver.o ./Drivers/Middleware/ExtMem/NOR_SFDP/stm32_sfdp_driver.su

.PHONY: clean-Drivers-2f-Middleware-2f-ExtMem-2f-NOR_SFDP

