################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../../../Middlewares/ST/STM32_ExtMem_Manager/boot/stm32_boot_lrun.c 

OBJS += \
./Drivers/Middleware/ExtMem/Boot/stm32_boot_lrun.o 

C_DEPS += \
./Drivers/Middleware/ExtMem/Boot/stm32_boot_lrun.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/Middleware/ExtMem/Boot/stm32_boot_lrun.o: ../../../Middlewares/ST/STM32_ExtMem_Manager/boot/stm32_boot_lrun.c Drivers/Middleware/ExtMem/Boot/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-Middleware-2f-ExtMem-2f-Boot

clean-Drivers-2f-Middleware-2f-ExtMem-2f-Boot:
	-$(RM) ./Drivers/Middleware/ExtMem/Boot/stm32_boot_lrun.cyclo ./Drivers/Middleware/ExtMem/Boot/stm32_boot_lrun.d ./Drivers/Middleware/ExtMem/Boot/stm32_boot_lrun.o ./Drivers/Middleware/ExtMem/Boot/stm32_boot_lrun.su

.PHONY: clean-Drivers-2f-Middleware-2f-ExtMem-2f-Boot

