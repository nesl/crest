################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../../../Middlewares/ST/STM32_ExtMem_Manager/stm32_extmem.c 

OBJS += \
./Drivers/Middleware/ExtMem/stm32_extmem.o 

C_DEPS += \
./Drivers/Middleware/ExtMem/stm32_extmem.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/Middleware/ExtMem/stm32_extmem.o: ../../../Middlewares/ST/STM32_ExtMem_Manager/stm32_extmem.c Drivers/Middleware/ExtMem/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-Middleware-2f-ExtMem

clean-Drivers-2f-Middleware-2f-ExtMem:
	-$(RM) ./Drivers/Middleware/ExtMem/stm32_extmem.cyclo ./Drivers/Middleware/ExtMem/stm32_extmem.d ./Drivers/Middleware/ExtMem/stm32_extmem.o ./Drivers/Middleware/ExtMem/stm32_extmem.su

.PHONY: clean-Drivers-2f-Middleware-2f-ExtMem

