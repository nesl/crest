/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file         stm32n6xx_hal_msp.c
  * @brief        This file provides code for the MSP Initialization
  *               and de-Initialization codes.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  *
  * Local modifications for CREST:
  * Modifications Copyright (c) 2026 Joseph Zales.
  *
  * Upstream base:
  *   STM32CubeN6/Projects/NUCLEO-N657X0-Q/Templates/Template_FSBL_LRUN/
  *   Appli/Src/stm32n6xx_hal_msp.c
  *
  * Local changes:
  *   - Added RTC MSP initialization and de-initialization hooks.
  *   - Added RTC EXTI line configuration and RTC_S interrupt wiring for the
  *     cadenced wake path used by CREST HIL runs.
  *
  * Licensing note:
  *   - The underlying file remains subject to the upstream STMicroelectronics
  *     license terms; this notice covers local modifications only.
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN TD */

/* USER CODE END TD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN Define */

/* USER CODE END Define */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN Macro */

/* USER CODE END Macro */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
static EXTI_HandleTypeDef s_rtc_exti;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* External functions --------------------------------------------------------*/
/* USER CODE BEGIN ExternalFunctions */

/* USER CODE END ExternalFunctions */

/* USER CODE BEGIN 0 */

/* USER CODE END 0 */
/**
  * Initializes the Global MSP.
  */
void HAL_MspInit(void)
{

  /* USER CODE BEGIN MspInit 0 */

  /* USER CODE END MspInit 0 */

  /* System interrupt init*/

  HAL_PWREx_EnableVddIO3();

  /* USER CODE BEGIN MspInit 1 */

  /* USER CODE END MspInit 1 */
}

/* USER CODE BEGIN 1 */

void HAL_RTC_MspInit(RTC_HandleTypeDef *hrtc)
{
  EXTI_ConfigTypeDef exti_config = {0};

  if (hrtc->Instance != RTC)
  {
    return;
  }

  __HAL_RCC_RTCAPB_CLK_ENABLE();
  __HAL_RCC_RTC_CLK_ENABLE();
  __HAL_RCC_RTC_CLK_SLEEP_ENABLE();
  __HAL_RCC_RTCAPB_CLK_SLEEP_ENABLE();

  s_rtc_exti.Line = EXTI_CONFIG;
  exti_config.Line = EXTI_LINE_17;
  exti_config.Mode = EXTI_MODE_INTERRUPT;
  exti_config.Trigger = EXTI_TRIGGER_RISING;
  HAL_EXTI_SetConfigLine(&s_rtc_exti, &exti_config);

  HAL_NVIC_SetPriority(RTC_S_IRQn, 0U, 0U);
  HAL_NVIC_EnableIRQ(RTC_S_IRQn);
}

void HAL_RTC_MspDeInit(RTC_HandleTypeDef *hrtc)
{
  if (hrtc->Instance != RTC)
  {
    return;
  }

  HAL_NVIC_DisableIRQ(RTC_S_IRQn);
  __HAL_RCC_RTCAPB_CLK_DISABLE();
  __HAL_RCC_RTC_CLK_DISABLE();
}

/* USER CODE END 1 */
